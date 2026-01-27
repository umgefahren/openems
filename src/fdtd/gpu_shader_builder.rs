//! GPU Shader Builder for dynamic extension compilation.
//!
//! This module provides a framework for composing WGSL shaders with extension
//! hooks, allowing GPU-native implementations of FDTD extensions like PML,
//! Mur ABC, and dispersive materials.

use crate::extensions::GpuExtensionData;

/// Shader hook points where extensions can inject code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderHook {
    /// Before H-field update (inside update_h kernel)
    PreUpdateH,
    /// After H-field update (inside update_h kernel)
    PostUpdateH,
    /// Before E-field update (inside update_e kernel)
    PreUpdateE,
    /// After E-field update (inside update_e kernel)
    PostUpdateE,
    /// Global declarations (buffer bindings, structs, functions)
    Declarations,
}

impl ShaderHook {
    /// Get the placeholder marker for this hook in the shader template.
    pub fn marker(&self) -> &'static str {
        match self {
            ShaderHook::PreUpdateH => "// #EXTENSION_PRE_UPDATE_H#",
            ShaderHook::PostUpdateH => "// #EXTENSION_POST_UPDATE_H#",
            ShaderHook::PreUpdateE => "// #EXTENSION_PRE_UPDATE_E#",
            ShaderHook::PostUpdateE => "// #EXTENSION_POST_UPDATE_E#",
            ShaderHook::Declarations => "// #EXTENSION_DECLARATIONS#",
        }
    }
}

/// Compiled extension shader code for a specific hook.
#[derive(Debug, Clone)]
pub struct ExtensionShaderCode {
    /// Which hook this code should be injected at.
    pub hook: ShaderHook,
    /// The WGSL code to inject.
    pub code: String,
}

/// Builder for composing WGSL shaders with extension hooks.
pub struct GpuShaderBuilder {
    /// Base shader template with hook placeholders.
    base_shader: String,
    /// Extension code to inject at each hook point.
    extension_code: Vec<ExtensionShaderCode>,
    /// Next available binding index for extension buffers.
    next_binding: u32,
}

impl GpuShaderBuilder {
    /// Create a new shader builder with the base FDTD shader template.
    pub fn new() -> Self {
        Self {
            base_shader: Self::base_shader_template(),
            extension_code: Vec::new(),
            next_binding: 11, // Start after core bindings (0-10)
        }
    }

    /// Get the base shader template with hook placeholders.
    fn base_shader_template() -> String {
        // Include the base shader and add hook placeholders
        let base = include_str!("shaders.wgsl");

        // The base shader already has the core FDTD logic.
        // We add hook markers at strategic points for extension injection.
        let mut shader = String::with_capacity(base.len() + 1000);

        // Add extension declarations section at the top
        shader.push_str("// Extension declarations\n");
        shader.push_str(ShaderHook::Declarations.marker());
        shader.push_str("\n\n");

        // Add the base shader
        shader.push_str(base);

        shader
    }

    /// Add extension shader code for a specific hook.
    pub fn add_extension_code(&mut self, hook: ShaderHook, code: String) {
        self.extension_code.push(ExtensionShaderCode { hook, code });
    }

    /// Add an extension using its GpuExtensionData.
    ///
    /// Returns the starting binding index used for this extension's buffers.
    pub fn add_extension(&mut self, data: &GpuExtensionData) -> u32 {
        let start_binding = self.next_binding;

        // Add shader code to declarations
        self.add_extension_code(ShaderHook::Declarations, data.shader_code.clone());

        // Update next binding based on extension's buffer count
        self.next_binding += data.buffers.len() as u32;

        start_binding
    }

    /// Compile the final shader with all extensions injected.
    pub fn build(&self) -> String {
        let mut shader = self.base_shader.clone();

        // Group extension code by hook
        let mut hook_code: std::collections::HashMap<ShaderHook, Vec<String>> =
            std::collections::HashMap::new();

        for ext in &self.extension_code {
            hook_code
                .entry(ext.hook)
                .or_insert_with(Vec::new)
                .push(ext.code.clone());
        }

        // Inject code at each hook point
        for hook in [
            ShaderHook::Declarations,
            ShaderHook::PreUpdateH,
            ShaderHook::PostUpdateH,
            ShaderHook::PreUpdateE,
            ShaderHook::PostUpdateE,
        ] {
            let marker = hook.marker();
            let replacement: String = hook_code
                .get(&hook)
                .map(|codes: &Vec<String>| codes.join("\n"))
                .unwrap_or_default();

            shader = shader.replace(marker, &replacement);
        }

        shader
    }

    /// Get the next available binding index.
    pub fn next_binding(&self) -> u32 {
        self.next_binding
    }
}

impl Default for GpuShaderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// WGSL code generator for common extension patterns.
pub struct WgslCodeGen;

impl WgslCodeGen {
    /// Generate buffer binding declaration.
    pub fn buffer_binding(binding: u32, name: &str, element_type: &str, read_only: bool) -> String {
        let access = if read_only { "read" } else { "read_write" };
        format!(
            "@group(0) @binding({}) var<storage, {}> {}: array<{}>;\n",
            binding, access, name, element_type
        )
    }

    /// Generate uniform buffer binding declaration.
    pub fn uniform_binding(binding: u32, name: &str, struct_name: &str) -> String {
        format!(
            "@group(0) @binding({}) var<uniform> {}: {};\n",
            binding, name, struct_name
        )
    }

    /// Generate a struct definition.
    pub fn struct_def(name: &str, fields: &[(&str, &str)]) -> String {
        let mut s = format!("struct {} {{\n", name);
        for (field_name, field_type) in fields {
            s.push_str(&format!("    {}: {},\n", field_name, field_type));
        }
        s.push_str("}\n");
        s
    }

    /// Generate a function call.
    pub fn function_call(name: &str, args: &[&str]) -> String {
        format!("{}({});\n", name, args.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_builder_creation() {
        let builder = GpuShaderBuilder::new();
        assert_eq!(builder.next_binding(), 11);
    }

    #[test]
    fn test_shader_hook_markers() {
        assert_eq!(
            ShaderHook::PreUpdateH.marker(),
            "// #EXTENSION_PRE_UPDATE_H#"
        );
        assert_eq!(
            ShaderHook::Declarations.marker(),
            "// #EXTENSION_DECLARATIONS#"
        );
    }

    #[test]
    fn test_shader_build_with_extension() {
        let mut builder = GpuShaderBuilder::new();

        builder.add_extension_code(
            ShaderHook::Declarations,
            "// Test extension declaration\nvar<private> test_var: f32;".to_string(),
        );

        let shader = builder.build();

        assert!(shader.contains("// Test extension declaration"));
        assert!(shader.contains("var<private> test_var: f32"));
    }

    #[test]
    fn test_wgsl_codegen_buffer() {
        let code = WgslCodeGen::buffer_binding(15, "pml_flux", "f32", false);
        assert!(code.contains("@group(0) @binding(15)"));
        assert!(code.contains("var<storage, read_write>"));
        assert!(code.contains("pml_flux"));
    }

    #[test]
    fn test_wgsl_codegen_struct() {
        let code = WgslCodeGen::struct_def("PmlData", &[("sigma", "f32"), ("kappa", "f32")]);
        assert!(code.contains("struct PmlData"));
        assert!(code.contains("sigma: f32"));
        assert!(code.contains("kappa: f32"));
    }
}
