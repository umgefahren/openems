// Pipeline constants for grid dimensions
override nx: u32;
override ny: u32;
override nz: u32;

// Fields: [Ex, Ey, Ez] (concatenated)
@group(0) @binding(0) var<storage, read_write> e_field: array<f32>;

// Fields: [Hx, Hy, Hz] (concatenated)
@group(0) @binding(1) var<storage, read_write> h_field: array<f32>;

// E Coefficients (Interleaved per cell: [Ca_x, Ca_y, Ca_z, Cb_x], [Cb_y, Cb_z, 0, 0])
@group(0) @binding(2) var<storage, read> e_coeff_f16: array<vec4<float16>>;
@group(0) @binding(3) var<storage, read> e_coeff_f32: array<vec4<f32>>;

// Cell Class Map
@group(0) @binding(4) var<storage, read> cell_class: array<u32>;

// H Coefficients (Interleaved per cell: [Da_x, Da_y, Da_z, Db_x], [Db_y, Db_z, 0, 0])
@group(0) @binding(5) var<storage, read> h_coeff_f16: array<vec4<float16>>;
@group(0) @binding(6) var<storage, read> h_coeff_f32: array<vec4<f32>>;

// Excitation data
struct ExcitationPoint {
    position: vec3<u32>,  // (i, j, k)
    direction: u32,        // 0=x, 1=y, 2=z
    value: f32,            // Field value to apply
    soft_source: u32,      // 0=hard, 1=soft
    _padding: vec2<u32>,   // Padding to align to 32 bytes
}

@group(0) @binding(7) var<storage, read> excitations: array<ExcitationPoint>;
@group(0) @binding(8) var<storage, read> excitation_offsets: array<u32>;  // Start index for each timestep

// Timestep uniform (updated each step via CPU write)
struct TimestepUniform {
    current_timestep: u32,
}
@group(0) @binding(9) var<uniform> timestep_data: TimestepUniform;

// Energy reduction output buffer
@group(0) @binding(10) var<storage, read_write> energy_output: array<f32>;

fn get_idx(x: u32, y: u32, z: u32) -> u32 {
    return x * ny * nz + y * nz + z;
}

struct Coeffs {
    c0: vec4<f32>,
    c1: vec4<f32>,
}

fn load_e_coeffs(idx: u32) -> Coeffs {
    let class_id = cell_class[idx];
    let base = idx * 2u;
    if (class_id == 0u) {
        return Coeffs(
            vec4<f32>(e_coeff_f16[base]),
            vec4<f32>(e_coeff_f16[base + 1u])
        );
    } else {
        return Coeffs(
            e_coeff_f32[base],
            e_coeff_f32[base + 1u]
        );
    }
}

fn load_h_coeffs(idx: u32) -> Coeffs {
    let class_id = cell_class[idx];
    let base = idx * 2u;
    if (class_id == 0u) {
        return Coeffs(
            vec4<f32>(h_coeff_f16[base]),
            vec4<f32>(h_coeff_f16[base + 1u])
        );
    } else {
        return Coeffs(
            h_coeff_f32[base],
            h_coeff_f32[base + 1u]
        );
    }
}

// ILP=4: Each thread processes 4 elements along K (Z-axis).
// Workgroup (64, 1, 1) -> processes 256 elements along Z per group.
@compute @workgroup_size(64, 1, 1)
fn update_h(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // k_base is the starting Z-index for this thread
    let k_base = global_id.x * 4u;
    let j = global_id.y;
    let i = global_id.z;

    if (i >= nx || j >= ny || k_base >= nz) {
        return;
    }

    let total = nx * ny * nz;
    let idx_base = get_idx(i, j, k_base);

    // Pre-load data for register reuse along Z-axis (neighbor optimization)
    // We need Ey at k, k+1, k+2, k+3, k+4 for 4 updates.
    
    // Load Ey window
    let ey_idx_base = 1u * total + idx_base;
    
    var ey_vals: array<f32, 5>;
    for (var u = 0u; u < 5u; u++) {
        let k_curr = k_base + u;
        if (k_curr < nz) {
            ey_vals[u] = e_field[ey_idx_base + u];
        } else {
            ey_vals[u] = 0.0;
        }
    }

    // Process 4 elements
    for (var u = 0u; u < 4u; u++) {
        let k = k_base + u;
        if (k >= nz) { break; }
        
        let idx = idx_base + u;
        
        // Load coefficients for this cell
        let coeffs = load_h_coeffs(idx);
        // c0: [da_x, da_y, da_z, db_x]
        // c1: [db_y, db_z, 0, 0]
        let da_x = coeffs.c0.x;
        let da_y = coeffs.c0.y;
        let da_z = coeffs.c0.z;
        let db_x = coeffs.c0.w;
        let db_y = coeffs.c1.x;
        let db_z = coeffs.c1.y;

        // Hx update: curl_x = dEz/dy - dEy/dz
        // dez_dy = Ez(i, j+1, k) - Ez(i, j, k)
        // dey_dz = Ey(i, j, k+1) - Ey(i, j, k)
        
        let ez_curr = e_field[2u * total + idx];
        let ez_jp1_val = select(0.0, e_field[2u * total + get_idx(i, j + 1u, k)] - ez_curr, j + 1u < ny);
        let dez_dy = ez_jp1_val;

        // Reuse Ey values from registers, but need boundary check
        let dey_dz = select(0.0, ey_vals[u + 1u] - ey_vals[u], k + 1u < nz);

        let curl_x = dez_dy - dey_dz;
        let hx_idx = 0u * total + idx;
        h_field[hx_idx] = fma(db_x, curl_x, da_x * h_field[hx_idx]);

        // Hy update: curl_y = dEx/dz - dEz/dx
        // dex_dz = Ex(i, j, k+1) - Ex(i, j, k)
        let ex_curr = e_field[0u * total + idx];
        let ex_kp1 = select(0.0, e_field[0u * total + idx + 1u] - ex_curr, k + 1u < nz);
        let dex_dz = ex_kp1;
        
        let dez_dx = select(0.0, e_field[2u * total + get_idx(i + 1u, j, k)] - ez_curr, i + 1u < nx);
        let curl_y = dex_dz - dez_dx;
        
        let hy_idx = 1u * total + idx;
        h_field[hy_idx] = fma(db_y, curl_y, da_y * h_field[hy_idx]);

        // Hz update: curl_z = dEy/dx - dEx/dy
        let dey_dx = select(0.0, e_field[1u * total + get_idx(i + 1u, j, k)] - ey_vals[u], i + 1u < nx);
        let dex_dy = select(0.0, e_field[0u * total + get_idx(i, j + 1u, k)] - ex_curr, j + 1u < ny);
        let curl_z = dey_dx - dex_dy;
        
        let hz_idx = 2u * total + idx;
        h_field[hz_idx] = fma(db_z, curl_z, da_z * h_field[hz_idx]);
    }
}

@compute @workgroup_size(64, 1, 1)
fn update_e(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let k_base = global_id.x * 4u;
    let j = global_id.y;
    let i = global_id.z;
    let total = nx * ny * nz;

    // Boundary check for main FDTD update
    if (i < 1u || i >= nx || j < 1u || j >= ny || k_base >= nz) {
        return;
    }

    let idx_base = get_idx(i, j, k_base);

    // Pre-load Hy for register reuse
    let hy_idx_base = 1u * total + idx_base;
    var hy_vals: array<f32, 5>; // Stores Hy[k-1], Hy[k], Hy[k+1], Hy[k+2], Hy[k+3]
    
    for (var u = 0u; u < 5u; u++) {
        let k_rel = k_base + u; 
        if (k_rel > 0u && k_rel <= nz + 1u) { // Rough check
             let read_k = k_rel - 1u;
             if (read_k < nz) {
                 hy_vals[u] = h_field[hy_idx_base + u - 1u];
             } else {
                 hy_vals[u] = 0.0;
             }
        } else {
             hy_vals[u] = 0.0;
        }
    }
    
    // Process 4 elements
    for (var u = 0u; u < 4u; u++) {
        let k = k_base + u;
        if (k < 1u || k >= nz) { continue; }
        
        let idx = idx_base + u;
        
        // Load coefficients for this cell
        let coeffs = load_e_coeffs(idx);
        // c0: [ca_x, ca_y, ca_z, cb_x]
        // c1: [cb_y, cb_z, 0, 0]
        let ca_x = coeffs.c0.x;
        let ca_y = coeffs.c0.y;
        let ca_z = coeffs.c0.z;
        let cb_x = coeffs.c0.w;
        let cb_y = coeffs.c1.x;
        let cb_z = coeffs.c1.y;

        // Ex update: curl_x = (Hz(j) - Hz(j-1)) - (Hy(k) - Hy(k-1))
        let hz_curr = h_field[2u * total + idx];
        let hz_jm1 = h_field[2u * total + get_idx(i, j - 1u, k)];
        
        let hy_k = hy_vals[u + 1u];
        let hy_km1 = hy_vals[u];
        
        let curl_x = (hz_curr - hz_jm1) - (hy_k - hy_km1);
        
        let ex_idx = 0u * total + idx;
        e_field[ex_idx] = fma(cb_x, curl_x, ca_x * e_field[ex_idx]);

        // Ey update: curl_y = (Hx(k) - Hx(k-1)) - (Hz(i) - Hz(i-1))
        let hx_curr = h_field[0u * total + idx];
        let hx_km1 = h_field[0u * total + idx - 1u]; // Safe since k>=1
        let hz_im1 = h_field[2u * total + get_idx(i - 1u, j, k)];
        
        let curl_y = (hx_curr - hx_km1) - (hz_curr - hz_im1);
        
        let ey_idx = 1u * total + idx;
        e_field[ey_idx] = fma(cb_y, curl_y, ca_y * e_field[ey_idx]);

        // Ez update: curl_z = (Hy(i) - Hy(i-1)) - (Hx(j) - Hx(j-1))
        let hy_im1 = h_field[1u * total + get_idx(i - 1u, j, k)];
        let hx_jm1 = h_field[0u * total + get_idx(i, j - 1u, k)];
        
        let curl_z = (hy_k - hy_im1) - (hx_curr - hx_jm1);
        
        let ez_idx = 2u * total + idx;
        e_field[ez_idx] = fma(cb_z, curl_z, ca_z * e_field[ez_idx]);
    }
}

// Energy reduction compute shader
// Uses workgroup reduction with Kahan summation for numerical stability.
// Kahan summation tracks and compensates for rounding errors, reducing
// error from O(n*eps) to O(eps) where eps is machine epsilon.
var<workgroup> e_partial: array<f32, 64>;
var<workgroup> h_partial: array<f32, 64>;

// Kahan summation: add value to sum while tracking compensation
// Returns the new sum; updates compensation in-place
fn kahan_add(sum: f32, value: f32, compensation: ptr<function, f32>) -> f32 {
    let y = value - *compensation;        // Compensated value
    let t = sum + y;                       // New sum (with rounding error)
    *compensation = (t - sum) - y;         // Recover rounding error for next iteration
    return t;
}

@compute @workgroup_size(64, 1, 1)
fn compute_energy(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) wg_id: vec3<u32>,
                  @builtin(num_workgroups) num_wgs: vec3<u32>) {
    let total = nx * ny * nz;
    let total_threads = num_wgs.x * 64u;
    let global_thread_id = wg_id.x * 64u + local_id.x;

    // Each thread computes partial sum using Kahan summation
    var e_sum: f32 = 0.0;
    var h_sum: f32 = 0.0;
    var e_comp: f32 = 0.0;  // Kahan compensation for E
    var h_comp: f32 = 0.0;  // Kahan compensation for H

    // Stride through field data
    var idx = global_thread_id;
    while (idx < total * 3u) {
        let e_val = e_field[idx];
        let h_val = h_field[idx];
        e_sum = kahan_add(e_sum, e_val * e_val, &e_comp);
        h_sum = kahan_add(h_sum, h_val * h_val, &h_comp);
        idx += total_threads;
    }

    // Store in workgroup shared memory
    e_partial[local_id.x] = e_sum;
    h_partial[local_id.x] = h_sum;
    workgroupBarrier();

    // Reduction within workgroup (sequential addressing)
    // Note: Using simple addition for reduction phase as values are similar magnitude
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if (local_id.x < stride) {
            e_partial[local_id.x] += e_partial[local_id.x + stride];
            h_partial[local_id.x] += h_partial[local_id.x + stride];
        }
        workgroupBarrier();
    }

    // First thread writes workgroup result
    if (local_id.x == 0u) {
        energy_output[wg_id.x * 2u] = e_partial[0];
        energy_output[wg_id.x * 2u + 1u] = h_partial[0];
    }
}

// Apply excitations to E-field
// CRITICAL: Must be called AFTER update_e to match BasicEngine behavior
// Single-threaded to avoid race conditions
@compute @workgroup_size(1, 1, 1)
fn apply_excitations(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Only thread (0,0,0) does work
    if (global_id.x != 0u || global_id.y != 0u || global_id.z != 0u) {
        return;
    }

    let total = nx * ny * nz;
    let ts = timestep_data.current_timestep;
    let exc_start = excitation_offsets[ts];
    let exc_end = excitation_offsets[ts + 1u];

    for (var exc_idx = exc_start; exc_idx < exc_end; exc_idx++) {
        let exc = excitations[exc_idx];
        let exc_linear_idx = get_idx(exc.position.x, exc.position.y, exc.position.z);
        let field_idx = exc.direction * total + exc_linear_idx;

        if (exc.soft_source != 0u) {
            e_field[field_idx] += exc.value;
        } else {
            e_field[field_idx] = exc.value;
        }
    }
}
