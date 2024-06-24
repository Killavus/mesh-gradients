use std::time::SystemTime;

use macroquad::prelude::*;
use macroquad::ui::{hash, root_ui};
use miniquad::window::set_window_size;
use nalgebra::{self as na, SimdPartialOrd};
use nalgebra::{matrix, vector};

#[derive(Debug)]
struct Mesh {
    width: usize,
    height: usize,
    points: Vec<ControlPoint>,
}

#[derive(Debug)]
struct ControlPoint {
    position: na::Vector2<f32>,
    u_tangent: na::Vector2<f32>,
    v_tangent: na::Vector2<f32>,
    color: na::Vector3<f32>,
}

impl ControlPoint {
    fn new(
        position: na::Vector2<f32>,
        color: na::Vector3<f32>,
        grid_w: usize,
        grid_h: usize,
    ) -> Self {
        let u_tangent = na::Vector2::new(2.0 / (grid_w - 1) as f32, 0.0) * 0.5;
        let v_tangent = na::Vector2::new(0.0, 2.0 / (grid_h - 1) as f32) * 0.5;

        Self {
            position,
            u_tangent,
            v_tangent,
            color,
        }
    }
}

impl Mesh {
    fn new(width: usize, height: usize, colors: Vec<na::Vector3<f32>>) -> Self {
        let mut count = 0;
        let x_step = 1.0 / (width - 1) as f32;
        let y_step = 1.0 / (height - 1) as f32;

        let points = std::iter::from_fn(move || {
            if count >= width * height {
                None
            } else {
                let x = (count % width) as f32 * x_step;
                let y = (count / width) as f32 * y_step;

                let result = Some(ControlPoint::new(
                    na::Vector2::new(x, y),
                    colors[count as usize],
                    width,
                    height,
                ));
                count += 1;
                result
            }
        });

        Self {
            width,
            height,
            points: points.collect(),
        }
    }

    fn point_at(&self, w: usize, h: usize) -> &ControlPoint {
        &self.points[(h * self.width + w) as usize]
    }
}

fn point_idx(mouse_pos: na::Vector2<f32>, mesh: &Mesh) -> Option<usize> {
    if mouse_pos.x > WORKSPACE_SIZE_W || mouse_pos.y > WORKSPACE_SIZE_H {
        None
    } else {
        for (idx, point) in mesh.points.iter().enumerate() {
            let spoint = ws_coord(&point.position);

            if (spoint - mouse_pos).norm() < 5.0 {
                return Some(idx);
            }
        }

        None
    }
}

const UI_SIZE: f32 = 200.0;
const WORKSPACE_SIZE_W: f32 = 600.0;
const WORKSPACE_SIZE_H: f32 = 600.0;
const WORKSPACE_PADDING: f32 = 160.0;

fn ws_coord(point: &na::Vector2<f32>) -> na::Vector2<f32> {
    let sw = WORKSPACE_SIZE_W - WORKSPACE_PADDING;
    let sh = WORKSPACE_SIZE_H - WORKSPACE_PADDING;
    point.component_mul(&na::Vector2::new(sw, sh))
        + na::Vector2::new(WORKSPACE_PADDING / 2.0, WORKSPACE_PADDING / 2.0)
}

fn pt_coord(point: &na::Vector2<f32>) -> na::Vector2<f32> {
    let sw = WORKSPACE_SIZE_W - WORKSPACE_PADDING;
    let sh = WORKSPACE_SIZE_H - WORKSPACE_PADDING;

    point
        .component_div(&na::Vector2::new(sw, sh))
        .simd_clamp(na::Vector2::new(0.0, 0.0), na::Vector2::new(1.0, 1.0))
}

const H: na::Matrix4<f32> = matrix![
     2.0, -3.0,  0.0,  1.0;
    -2.0,  3.0,  0.0,  0.0;
     1.0, -2.0,  1.0,  0.0;
     1.0, -1.0,  0.0,  0.0;
];

fn cubic_colvec(v: f32) -> na::Vector4<f32> {
    vector![v * v * v, v * v, v, 1.0]
}

#[derive(PartialEq, Eq)]
enum Axis {
    X,
    Y,
}

#[derive(PartialEq, Eq)]
enum ColorAxis {
    R,
    G,
    B,
}

fn geometric_coefficients(
    p00: &ControlPoint,
    p01: &ControlPoint,
    p10: &ControlPoint,
    p11: &ControlPoint,
    axis: Axis,
) -> na::Matrix4<f32> {
    let l = |p: &ControlPoint| match axis {
        Axis::X => p.position.x,
        Axis::Y => p.position.y,
    };

    let u = |p: &ControlPoint| match axis {
        Axis::X => p.u_tangent.x,
        Axis::Y => p.u_tangent.y,
    };

    let v = |p: &ControlPoint| match axis {
        Axis::X => p.v_tangent.x,
        Axis::Y => p.v_tangent.y,
    };

    matrix![
        l(p00), l(p01), v(p00), v(p01);
        l(p10), l(p11), v(p10), v(p11);
        u(p00), u(p01), 0.0, 0.0;
        u(p10), u(p11), 0.0, 0.0;
    ]
    .transpose()
}

fn color_coefficients(
    p00: &ControlPoint,
    p01: &ControlPoint,
    p10: &ControlPoint,
    p11: &ControlPoint,
    color: ColorAxis,
) -> na::Matrix4<f32> {
    let l = |p: &ControlPoint| match color {
        ColorAxis::R => p.color.x,
        ColorAxis::G => p.color.y,
        ColorAxis::B => p.color.z,
    };

    matrix![
        l(p00), l(p01), 0.0, 0.0;
        l(p10), l(p11), 0.0, 0.0;
        0.0, 0.0, 0.0, 0.0;
        0.0, 0.0, 0.0, 0.0;
    ]
    .transpose()
}

fn ferguson_patch_pt(
    u: f32,
    v: f32,
    geom_x: &na::Matrix4<f32>,
    geom_y: &na::Matrix4<f32>,
) -> na::Vector2<f32> {
    let u_vec = cubic_colvec(u);
    let v_vec = cubic_colvec(v);

    let x_acc = H.transpose() * geom_x.transpose() * H;
    let y_acc = H.transpose() * geom_y.transpose() * H;

    let ux = x_acc * u_vec;
    let uy = y_acc * u_vec;

    na::Vector2::new(ux.dot(&v_vec), uy.dot(&v_vec))
}

fn ferguson_patch_col(
    u: f32,
    v: f32,
    rgb_coeffs: (&na::Matrix4<f32>, &na::Matrix4<f32>, &na::Matrix4<f32>),
) -> na::Vector3<f32> {
    let u_vec = cubic_colvec(u);
    let v_vec = cubic_colvec(v);

    let r_acc = H.transpose() * rgb_coeffs.0.transpose() * H;
    let g_acc = H.transpose() * rgb_coeffs.1.transpose() * H;
    let b_acc = H.transpose() * rgb_coeffs.2.transpose() * H;

    let ur = r_acc * u_vec;
    let ug = g_acc * u_vec;
    let ub = b_acc * u_vec;

    na::Vector3::new(ur.dot(&v_vec), ug.dot(&v_vec), ub.dot(&v_vec))
}

fn draw_across_ferguson_axis(
    geom_x: &na::Matrix4<f32>,
    geom_y: &na::Matrix4<f32>,
    rgb_coeffs: (&na::Matrix4<f32>, &na::Matrix4<f32>, &na::Matrix4<f32>),
    const_val: f32,
    steps: u32,
    axis: Axis,
) {
    let u = |t: f32| match axis {
        Axis::X => t,
        Axis::Y => const_val,
    };

    let v = |t| match axis {
        Axis::X => const_val,
        Axis::Y => t,
    };

    let mut last_point = ferguson_patch_pt(u(0.0), v(0.0), geom_x, geom_y);

    for i in 1..=steps {
        let point = ferguson_patch_pt(
            u(i as f32 / steps as f32),
            v(i as f32 / steps as f32),
            geom_x,
            geom_y,
        );

        let color = ferguson_patch_col(
            u(i as f32 / steps as f32),
            v(i as f32 / steps as f32),
            rgb_coeffs,
        );

        draw_line(
            ws_coord(&last_point).x,
            ws_coord(&last_point).y,
            ws_coord(&point).x,
            ws_coord(&point).y,
            2.0,
            Color::from_rgba(
                (&color.x * 255.0) as u8,
                (&color.y * 255.0) as u8,
                (&color.z * 255.0) as u8,
                255,
            ),
        );
        last_point = point;
    }
}

fn draw_hermite_from_geom(
    geom_x: &na::Matrix4<f32>,
    geom_y: &na::Matrix4<f32>,
    rgb_coeffs: (&na::Matrix4<f32>, &na::Matrix4<f32>, &na::Matrix4<f32>),
    steps: u32,
) {
    // top
    draw_across_ferguson_axis(geom_x, geom_y, rgb_coeffs, 0.0, steps, Axis::Y);
    // bottom
    draw_across_ferguson_axis(geom_x, geom_y, rgb_coeffs, 1.0, steps, Axis::Y);
    // leading
    draw_across_ferguson_axis(geom_x, geom_y, rgb_coeffs, 0.0, steps, Axis::X);
    // trailing
    draw_across_ferguson_axis(geom_x, geom_y, rgb_coeffs, 1.0, steps, Axis::X);

    for i in 0..20 {
        for j in 0..20 {
            let u = i as f32 / 20.0;
            let v = j as f32 / 20.0;

            let point = ferguson_patch_pt(u, v, geom_x, geom_y);
            let color = ferguson_patch_col(u, v, rgb_coeffs);

            draw_circle(
                ws_coord(&point).x,
                ws_coord(&point).y,
                2.0,
                Color::from_rgba(
                    (&color.x * 255.0) as u8,
                    (&color.y * 255.0) as u8,
                    (&color.z * 255.0) as u8,
                    255,
                ),
            );
        }
    }
}

fn construct_mesh(
    mesh: &Mesh,
    subdivs: usize,
) -> (Vec<na::Vector3<f32>>, Vec<na::Vector3<f32>>, Vec<u32>) {
    let col_len = (mesh.width - 1) * (subdivs + 2);
    let row_len = (mesh.height - 1) * (subdivs + 2);

    let entries = col_len * row_len;

    let mut positions = Vec::with_capacity(entries);
    let mut colors = Vec::with_capacity(entries);
    let mut indexes = Vec::with_capacity(entries * 3 * 2);

    // mesh with subdivs = 3
    //  0  1  2  3  4
    //  5  6  7  8  9
    // 10 11 12 13 14

    // indexes:
    // 5 1 0
    // 5 6 1
    // 7 2 1
    // 6 7 2
    // 7 3 2
    // 7 8 3
    // 8 4 3
    // 8 9 4
    // 10 6 5
    // 10 11 6
    // 11 7 6
    // 11 12 7
    // 12 8 7
    // 12 13 8
    // 13 9 8
    // 13 14 9

    for w in 0..mesh.width - 1 {
        for h in 0..mesh.height - 1 {
            let p00 = mesh.point_at(w, h);
            let p01 = mesh.point_at(w, h + 1);
            let p10 = mesh.point_at(w + 1, h);
            let p11 = mesh.point_at(w + 1, h + 1);

            let x_coeff = geometric_coefficients(p00, p01, p10, p11, Axis::X);
            let y_coeff = geometric_coefficients(p00, p01, p10, p11, Axis::Y);
            let r_coeff = color_coefficients(p00, p01, p10, p11, ColorAxis::R);
            let g_coeff = color_coefficients(p00, p01, p10, p11, ColorAxis::G);
            let b_coeff = color_coefficients(p00, p01, p10, p11, ColorAxis::B);

            let steps = subdivs + 1;
            let index_start = positions.len();

            for i in 0..=steps {
                for j in 0..=steps {
                    let u = i as f32 / steps as f32;
                    let v = j as f32 / steps as f32;

                    let point = {
                        let mut p = ferguson_patch_pt(u, v, &x_coeff, &y_coeff);
                        p *= 2.0;
                        p -= na::Vector2::new(1.0, 1.0);
                        p.component_mul_assign(&na::Vector2::new(1.0, -1.0));

                        na::Vector3::new(p.x, p.y, 0.0)
                    };

                    let color = ferguson_patch_col(u, v, (&r_coeff, &g_coeff, &b_coeff));

                    positions.push(point);
                    colors.push(color);
                }
            }

            let row_len = steps + 1;

            for r in 0..steps {
                for c in 0..steps {
                    indexes.push((index_start + r * row_len + c + row_len) as u32);
                    indexes.push((index_start + r * row_len + c + 1) as u32);
                    indexes.push((index_start + r * row_len + c) as u32);

                    indexes.push((index_start + r * row_len + c + row_len) as u32);
                    indexes.push((index_start + r * row_len + c + row_len + 1) as u32);
                    indexes.push((index_start + r * row_len + c + 1) as u32);
                }
            }
        }
    }

    (positions, colors, indexes)
}

#[macroquad::main("Mesh Gradient")]
async fn main() {
    #[rustfmt::skip]
    let mut mesh = Mesh::new(
        3,
        3,
        vec![
            vector![0.0, 0.0, 0.0], vector![0.0, 0.0, 0.0], vector![0.0, 0.0, 0.0],
            vector![0.0, 0.0, 1.0], vector![0.0, 0.0, 1.0], vector![0.0, 0.0, 1.0],
            vector![0.0, 1.0, 0.0], vector![0.0, 1.0, 0.0], vector![0.0, 1.0, 0.0]
        ],
    );

    set_window_size((WORKSPACE_SIZE_W + UI_SIZE) as u32, WORKSPACE_SIZE_H as u32);

    let mut active_point_idx: Option<usize> = None;
    let mut last_mouse_pos: Option<(f32, f32)> = None;

    let mut x_pos_text = String::new();
    let mut y_pos_text = String::new();
    let mut subdivs = 0.0;

    loop {
        clear_background(WHITE);

        for (idx, point) in mesh.points.iter().enumerate() {
            let spoint = ws_coord(&point.position);

            draw_circle_lines(
                spoint.x,
                spoint.y,
                5.0,
                3.0,
                Color::from_rgba(
                    (&point.color.x * 255.0) as u8,
                    (&point.color.y * 255.0) as u8,
                    (&point.color.z * 255.0) as u8,
                    255,
                ),
            );

            if let Some(active_point_idx) = active_point_idx {
                if active_point_idx == idx {
                    draw_circle_lines(spoint.x, spoint.y, 7.0, 2.0, RED);
                }
            }
        }

        root_ui().window(
            hash!(),
            vec2(WORKSPACE_SIZE_W, 0.0),
            vec2(UI_SIZE, WORKSPACE_SIZE_H),
            |ui| {
                if let Some(point_idx) = active_point_idx {
                    let point = &mesh.points[point_idx];

                    ui.label(None, &format!("x: {}", point.position.x));
                    ui.label(None, &format!("y: {}", point.position.y));

                    ui.editbox(hash!(), vec2(100.0, 20.0), &mut x_pos_text);
                    ui.editbox(hash!(), vec2(100.0, 20.0), &mut y_pos_text);

                    if ui.button(None, "Update point") {
                        let x = x_pos_text.parse::<f32>();
                        let y = y_pos_text.parse::<f32>();

                        if let Ok(x) = x {
                            mesh.points[point_idx].position.x = x;
                        }

                        if let Ok(y) = y {
                            mesh.points[point_idx].position.y = y;
                        }
                    }
                } else {
                    ui.label(None, "No point selected");
                }

                ui.separator();
                ui.slider(hash!(), "Subdivs", 0.0..20.0, &mut subdivs);
                ui.label(None, &format!("Subdivs: {}", subdivs.floor()));
                if ui.button(None, "Save mesh") {
                    let (positions, colors, indexes) =
                        construct_mesh(&mesh, subdivs.floor() as usize);

                    let json = serde_json::json!(
                        {
                            "positions": positions,
                            "colors": colors,
                            "indexes": indexes
                        }
                    );

                    serde_json::to_writer(
                        std::fs::File::create(format!(
                            "mesh-{}-subdiv{}.json",
                            SystemTime::now()
                                .duration_since(SystemTime::UNIX_EPOCH)
                                .unwrap()
                                .as_secs(),
                            subdivs.floor() as usize
                        ))
                        .unwrap(),
                        &json,
                    )
                    .unwrap();
                }
            },
        );

        if is_mouse_button_down(MouseButton::Left) {
            let mouse_pos = mouse_position();

            if mouse_pos.0 < WORKSPACE_SIZE_W {
                if let Some(last_mouse_pos) = last_mouse_pos {
                    let last_mouse_pos = na::Vector2::new(last_mouse_pos.0, last_mouse_pos.1);
                    let mouse_pos = na::Vector2::new(mouse_pos.0, mouse_pos.1);

                    if let Some(active_point_idx) = active_point_idx {
                        let delta = pt_coord(&(mouse_pos - last_mouse_pos));
                        mesh.points[active_point_idx].position += delta;
                    }
                }

                active_point_idx = point_idx(na::Vector2::new(mouse_pos.0, mouse_pos.1), &mesh);
                last_mouse_pos = Some(mouse_pos);
            }
        }

        if is_mouse_button_released(MouseButton::Left) {
            last_mouse_pos = None;
        }

        for w in 0..mesh.width - 1 {
            for h in 0..mesh.height - 1 {
                let p00 = mesh.point_at(w, h);
                let p01 = mesh.point_at(w, h + 1);
                let p10 = mesh.point_at(w + 1, h);
                let p11 = mesh.point_at(w + 1, h + 1);

                let x_coeff = geometric_coefficients(p00, p01, p10, p11, Axis::X);
                let y_coeff = geometric_coefficients(p00, p01, p10, p11, Axis::Y);
                let r_coeff = color_coefficients(p00, p01, p10, p11, ColorAxis::R);
                let g_coeff = color_coefficients(p00, p01, p10, p11, ColorAxis::G);
                let b_coeff = color_coefficients(p00, p01, p10, p11, ColorAxis::B);

                draw_hermite_from_geom(&x_coeff, &y_coeff, (&r_coeff, &g_coeff, &b_coeff), 100);
            }
        }

        next_frame().await;
    }
}
