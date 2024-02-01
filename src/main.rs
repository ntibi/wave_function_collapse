use rand::{rngs, seq::SliceRandom, thread_rng, Rng, SeedableRng};
use std::collections::{HashMap, VecDeque};

use bevy::{
    input::common_conditions::input_just_pressed, prelude::*, render::camera::ScalingMode,
    sprite::MaterialMesh2dBundle,
};

const TILESIZE: f32 = 32.;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TileContent(u16);

// TODO for a nicer map
// make virtual tiles
// split walls into leftwall, rightwall, upwall, downwall, topleftwall, toprightwall, bottomleftwall, bottomrightwall
//   they render to the same tile
bitflags::bitflags! {
    impl TileContent: u16 {
        const Grass = 1 << 0;
        const Dirt = 1 << 1;
        const Rock = 1 << 2;
        const Water = 1 << 3;

        // TODO cleaner way to set this ?
        const ALL = (1 << 4) - 1;
        const NONE = 0;
    }
}

#[derive(Resource)]
struct Wfc {
    w: usize,
    h: usize,

    tiles: Vec<TileContent>,
    wave: VecDeque<UVec2>,

    rules: Rules,
    weights: Weights,

    rng: rngs::StdRng,

    /// highlight the tile that got forced to collapse
    highlighted_forced_collapse: Option<UVec2>,
    /// highlight the tiles that got updated and collapsed
    highlight_updated_collapsed: Vec<UVec2>,
    /// highlight the tiles that got updated
    highlight_updated: Vec<UVec2>,

    // refs to entities for easier existence checks/deletions
    entities: Vec<Option<Entity>>,
    children: Vec<Vec<Entity>>,
}

#[derive(Component)]
struct Tile;

#[derive(Clone, Copy, Debug)]
enum Direction {
    Left = 0,
    Right,
    Up,
    Down,
}

impl Direction {
    fn opposite(&self) -> Direction {
        match self {
            Self::Left => Self::Right,
            Self::Right => Self::Left,
            Self::Up => Self::Down,
            Self::Down => Self::Up,
        }
    }
}

struct Rules {
    rules: HashMap<TileContent, [TileContent; 4]>,
}

impl Rules {
    fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }

    /// * `tile` - which tile is it (or set of potential tiles)
    /// * `dir` - position from the tile (from the tile's pov)
    fn get_allowed_states(&self, tile: TileContent, dir: Direction) -> TileContent {
        let mut allowed_states = TileContent::NONE;

        tile.iter().for_each(|state| {
            if let Some(rules) = self.rules.get(&state) {
                allowed_states |= rules[dir as usize];
            }
        });
        allowed_states
    }

    fn add_rule(&mut self, tile: TileContent, dir: Direction, allowed_states: TileContent) {
        assert!(tile.bits().count_ones() == 1);

        if let Some(rules) = self.rules.get_mut(&tile) {
            rules[dir as usize] |= allowed_states;
        } else {
            self.rules.insert(tile, [TileContent::NONE; 4]);
            self.rules.get_mut(&tile).unwrap()[dir as usize] = allowed_states;
        }
        // TODO set the rule both ways ?
    }
}

struct Weights {
    weights: HashMap<(TileContent, TileContent), f32>,
}

impl Weights {
    fn new() -> Self {
        Self {
            weights: HashMap::new(),
        }
    }

    #[allow(dead_code)]
    fn add_weight(&mut self, tile1: TileContent, tile2: TileContent, weight: f32) {
        assert!(tile1.bits().count_ones() == 1);
        assert!(tile2.bits().count_ones() == 1);
        assert!(self.weights.get(&(tile1, tile2)).is_none());
        assert!(self.weights.get(&(tile2, tile1)).is_none());

        self.weights.insert((tile1, tile2), weight);
        self.weights.insert((tile2, tile1), weight);
    }

    fn get_weight(&self, tile1: TileContent, tile2: TileContent) -> f32 {
        *self.weights.get(&(tile1, tile2)).unwrap_or(&1.)
    }
}

impl Wfc {
    fn new(w: usize, h: usize) -> Self {
        let mut wfc = Self {
            w,
            h,
            tiles: Vec::new(),
            entities: Vec::new(),
            children: Vec::new(),
            wave: VecDeque::new(),
            rules: Rules::new(),
            weights: Weights::new(),
            highlighted_forced_collapse: None,
            highlight_updated: Vec::new(),
            highlight_updated_collapsed: Vec::new(),
            rng: rngs::StdRng::seed_from_u64(1),
        };

        wfc.init();

        wfc
    }

    fn init(&mut self) {
        self.tiles = vec![TileContent::ALL; self.w * self.h];
        self.entities = vec![None; self.w * self.h];
        self.children = vec![Vec::new(); self.w * self.h];
        let seed = thread_rng().gen();
        println!("seed: {}", seed);
        self.rng = rngs::StdRng::seed_from_u64(seed);
        self.wave = VecDeque::new();
    }

    fn iter(&self) -> impl Iterator<Item = (UVec2, TileContent)> + '_ {
        self.tiles.iter().enumerate().map(move |(i, &content)| {
            (
                UVec2::new((i % self.w) as u32, (i / self.w) as u32),
                content,
            )
        })
    }

    fn get(&self, coords: UVec2) -> TileContent {
        self.tiles[coords.x as usize + coords.y as usize * self.w]
    }

    fn set(&mut self, coords: UVec2, content: TileContent) {
        self.tiles[coords.x as usize + coords.y as usize * self.w] = content;
    }

    /// gets the possible states for a tile according to its neighbours
    // TODO trick question
    // in this fn should i filter to only use the collapsed neighbours ?
    // or any neighbour is great bc it will allow more updates on the tiles
    // i dont really know, try both ?
    fn get_possible_states(&self, coords: UVec2) -> TileContent {
        let mut possible_states = TileContent::ALL;

        if coords.x > 0 {
            let neighbor = self.get(UVec2::new(coords.x - 1, coords.y));
            if neighbor.bits().count_ones() == 1 {
                possible_states &= self
                    .rules
                    .get_allowed_states(neighbor, Direction::Left.opposite());
            }
        }
        if (coords.x as usize) < self.w - 1 {
            let neighbor = self.get(UVec2::new(coords.x + 1, coords.y));
            if neighbor.bits().count_ones() == 1 {
                possible_states &= self
                    .rules
                    .get_allowed_states(neighbor, Direction::Right.opposite());
            }
        }
        if coords.y > 0 {
            let neighbor = self.get(UVec2::new(coords.x, coords.y - 1));
            if neighbor.bits().count_ones() == 1 {
                possible_states &= self
                    .rules
                    .get_allowed_states(neighbor, Direction::Up.opposite());
            }
        }
        if (coords.y as usize) < self.h - 1 {
            let neighbor = self.get(UVec2::new(coords.x, coords.y + 1));
            if neighbor.bits().count_ones() == 1 {
                possible_states &= self
                    .rules
                    .get_allowed_states(neighbor, Direction::Down.opposite());
            }
        }

        possible_states
    }

    fn collapse(&mut self, coords: UVec2) {
        let collapsed_neighbors = self
            .neighbours(coords)
            .filter_map(|(coords, dir)| {
                let neighbor = self.get(coords);
                if neighbor.bits().count_ones() == 1 {
                    Some((neighbor, dir))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let possible_states = collapsed_neighbors
            .iter()
            .fold(TileContent::ALL, |acc, &(neighbor, dir)| {
                acc & self.rules.get_allowed_states(neighbor, dir.opposite())
            });
        let state = possible_states.iter().collect::<Vec<_>>();
        let state = state
            .choose_weighted(&mut self.rng, |&state| {
                let v = collapsed_neighbors
                    .iter()
                    .map(|&(neighbor, _dir)| self.weights.get_weight(state, neighbor))
                    .sum::<f32>();
                if v > 0. {
                    v
                } else {
                    1.
                }
            })
            .unwrap();
        self.set(coords, *state);
    }

    fn neighbours(&self, coords: UVec2) -> impl Iterator<Item = (UVec2, Direction)> {
        let x = coords.x as u32;
        let y = coords.y as u32;

        let mut neighbours = Vec::new();
        if x > 0 {
            neighbours.push((UVec2::new(x - 1, y), Direction::Left));
        }
        if (x as usize) < self.w - 1 {
            neighbours.push((UVec2::new(x + 1, y), Direction::Right));
        }
        if y > 0 {
            neighbours.push((UVec2::new(x, y - 1), Direction::Up));
        }
        if (y as usize) < self.h - 1 {
            neighbours.push((UVec2::new(x, y + 1), Direction::Down));
        }

        neighbours.into_iter()
    }

    fn step(&mut self, n: usize) {
        self.highlighted_forced_collapse = None;
        self.highlight_updated.clear();
        self.highlight_updated_collapsed.clear();

        for _ in 0..n {
            if let Some(to_collapse) = self.wave.pop_front() {
                let possible_states = self.get(to_collapse);
                if possible_states.bits().count_ones() == 1 {
                    // already collapsed
                    continue;
                }
                let new_possible_states = self.get_possible_states(to_collapse);

                self.highlight_updated.push(to_collapse);
                if new_possible_states != possible_states {
                    self.set(to_collapse, new_possible_states);
                    self.highlight_updated_collapsed.push(to_collapse);
                    self.neighbours(to_collapse)
                        // TODO ?
                        //.filter(|&coords| self.get(coords).bits().count_ones() > 1)
                        .for_each(|(coords, _)| self.wave.push_back(coords));
                }
            } else {
                // get lowest entropy tiles
                // TODO these two steps (find lowest entropy and list tiles with lowest entropy) could be done in one pass
                let lowest_entropy = self
                    .tiles
                    .iter()
                    .filter_map(|&tile| {
                        let v = tile.bits().count_ones();
                        if v > 1 {
                            Some(v)
                        } else {
                            None
                        }
                    })
                    .min();
                if let Some(lowest_entropy) = lowest_entropy {
                    let lowest_entropy_tiles: Vec<UVec2> = self
                        .iter()
                        .filter_map(|(coords, tile)| {
                            if tile.bits().count_ones() == lowest_entropy {
                                Some(coords)
                            } else {
                                None
                            }
                        })
                        .collect();
                    let to_collapse = *lowest_entropy_tiles.choose(&mut self.rng).unwrap();
                    self.collapse(to_collapse);
                    self.highlighted_forced_collapse = Some(to_collapse);
                    self.neighbours(to_collapse)
                        // TODO ?
                        //.filter(|&coords| self.get(coords).bits().count_ones() > 1)
                        .for_each(|(coords, _)| self.wave.push_back(coords));
                } else {
                    // no entropy left, we're done
                    return;
                }
            }
        }
    }
}

#[derive(Resource)]
struct Stepping {
    steps_per_second: f32,
    remainder: f32,
}

#[derive(Resource)]
struct ForceReload(Option<UVec2>);

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                present_mode: bevy::window::PresentMode::AutoNoVsync,
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, (setup_camera, setup_rules, pause_time))
        .add_systems(Update, (step_wfc, update_wfc_tiles, debug_grid).chain())
        .add_systems(Update, (update_zoom, update_stepping, reset, update_mouse))
        .add_systems(
            PreUpdate,
            update_pause.run_if(input_just_pressed(KeyCode::Space)),
        )
        .insert_resource(Stepping {
            steps_per_second: 10.,
            remainder: 0.,
        })
        .insert_resource(ForceReload(Some(UVec2::new(0, 0))))
        .insert_resource(Wfc::new(32, 32))
        .run();
}

fn setup_camera(mut commands: Commands, wfc: Res<Wfc>) {
    let center = Vec2::new(wfc.w as f32 / 2. * TILESIZE, wfc.h as f32 / 2. * TILESIZE);

    commands.spawn((
        Camera2dBundle {
            transform: Transform::from_translation(center.extend(-100.)),
            camera: Camera {
                hdr: false,
                ..default()
            },
            projection: OrthographicProjection {
                near: -1000.,
                far: 1000.,
                scaling_mode: ScalingMode::AutoMax {
                    max_width: 1920.,
                    max_height: 1080.,
                },
                ..default()
            },
            ..default()
        },
        Name::new("camera"),
    ));
}

fn update_zoom(
    keyboard_input: Res<Input<KeyCode>>,
    mut camera_query: Query<&mut Transform, With<Camera>>,
    time: Res<Time<Real>>,
) {
    let mut camera_transform = camera_query.single_mut();
    let mut zoom = camera_transform.scale.x;
    if keyboard_input.pressed(KeyCode::Minus) {
        zoom += zoom * 1. * time.delta_seconds();
    }
    if keyboard_input.pressed(KeyCode::Equals) {
        zoom -= zoom * 1. * time.delta_seconds();
    }
    if keyboard_input.pressed(KeyCode::Back) {
        zoom = 1.;
    }
    camera_transform.scale = Vec2::splat(zoom).extend(1.);
}

fn update_wfc_tiles(
    mut commands: Commands,
    mut wfc: ResMut<Wfc>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    force_reload: Res<ForceReload>,
) {
    let mut entity_indices_to_flush: Vec<usize> = Vec::new();
    let mut entity_indices_to_update: Vec<(usize, Entity)> = Vec::new();
    let mut child_entities_indices_to_update: Vec<(usize, Vec<Entity>)> = Vec::new();

    for (i, (coords, tile)) in wfc.iter().enumerate() {
        let mut spawn = false;
        if let Some(entity) = wfc.entities[i] {
            let states = wfc.children[i].len();

            if Some(coords) == force_reload.0 || states != tile.bits().count_ones() as usize {
                commands.entity(entity).despawn_recursive();
                entity_indices_to_flush.push(i);
                spawn = true;
            }
        } else {
            spawn = true;
        }

        if spawn {
            let parent = commands
                .spawn((
                    SpatialBundle::from_transform(Transform::from_translation(Vec3::new(
                        coords.x as f32 * TILESIZE + TILESIZE / 2.,
                        coords.y as f32 * TILESIZE + TILESIZE / 2.,
                        0.,
                    ))),
                    Tile,
                ))
                .id();
            let n = tile.bits().count_ones();
            let states_per_axis = (n as f32).powf(0.5).ceil() as usize;
            let mut children = Vec::new();
            for (i, state) in tile.iter().enumerate() {
                let width = TILESIZE / states_per_axis as f32;
                let height = TILESIZE / states_per_axis as f32;
                let pos = Vec2::new(
                    width * (i % states_per_axis) as f32 + width / 2.,
                    height * (i / states_per_axis) as f32 + height / 2.,
                ) - TILESIZE / 2.;
                let child = commands
                    .spawn((MaterialMesh2dBundle {
                        mesh: meshes
                            .add(Mesh::from(shape::Quad::new(Vec2::new(width, height))))
                            .into(),
                        material: materials.add(ColorMaterial::from(match state {
                            TileContent::Grass => Color::rgb_u8(19, 133, 16),
                            TileContent::Dirt => Color::rgb_u8(118, 85, 43),
                            TileContent::Water => Color::rgb_u8(66, 141, 245),
                            TileContent::Rock => Color::rgb_u8(78, 81, 102),
                            _ => panic!("unhandled tile content {:?}", state),
                        })),

                        transform: Transform::from_translation(pos.extend(0.)),
                        ..default()
                    },))
                    .id();
                children.push(child);
            }
            commands.entity(parent).push_children(&children);
            entity_indices_to_update.push((i, parent));
            child_entities_indices_to_update.push((i, children));
        }
    }

    for i in entity_indices_to_flush {
        wfc.entities[i] = None;
        wfc.children[i] = vec![];
    }

    for (i, entity) in entity_indices_to_update {
        wfc.entities[i] = Some(entity);
    }
    for (i, children) in child_entities_indices_to_update {
        wfc.children[i] = children;
    }
}

fn debug_grid(wfc: Res<Wfc>, mut gizmos: Gizmos) {
    for (coords, _tile) in wfc.iter() {
        gizmos.rect_2d(
            coords.as_vec2() * TILESIZE + TILESIZE / 2.,
            0.,
            Vec2::splat(TILESIZE),
            Color::BLACK,
        );
    }
    for coords in &wfc.highlight_updated {
        gizmos.rect_2d(
            coords.as_vec2() * TILESIZE + TILESIZE / 2.,
            0.,
            Vec2::splat(TILESIZE),
            Color::ORANGE,
        );
    }
    for coords in &wfc.highlight_updated_collapsed {
        gizmos.rect_2d(
            coords.as_vec2() * TILESIZE + TILESIZE / 2.,
            0.,
            Vec2::splat(TILESIZE),
            Color::GREEN,
        );
    }
    if let Some(coords) = wfc.highlighted_forced_collapse {
        gizmos.rect_2d(
            coords.as_vec2() * TILESIZE + TILESIZE / 2.,
            0.,
            Vec2::splat(TILESIZE),
            Color::RED,
        );
    }
}

fn step_wfc(
    mut wfc: ResMut<Wfc>,
    mut stepping: ResMut<Stepping>,
    time: Res<Time>,
    keyboard_input: Res<Input<KeyCode>>,
) {
    if keyboard_input.just_pressed(KeyCode::Return) {
        wfc.step(1);
    } else {
        let steps = time.delta_seconds() * stepping.steps_per_second + stepping.remainder;
        stepping.remainder = steps.fract();
        let steps = steps.floor() as usize;
        if steps > 0 {
            wfc.step(steps);
        }
    }
}

fn update_pause(mut time: ResMut<Time<Virtual>>) {
    if time.is_paused() {
        time.unpause();
    } else {
        time.pause();
    }
}

fn update_stepping(mut stepping: ResMut<Stepping>, keyboard_input: Res<Input<KeyCode>>) {
    if keyboard_input.just_pressed(KeyCode::BracketRight) {
        stepping.steps_per_second *= 2.;
        println!("steps per second: {}", stepping.steps_per_second);
    }
    if keyboard_input.just_pressed(KeyCode::BracketLeft) {
        stepping.steps_per_second /= 2.;
        println!("steps per second: {}", stepping.steps_per_second);
    }
}

fn setup_rules(mut wfc: ResMut<Wfc>) {
    type C = TileContent;
    type D = Direction;

    wfc.rules
        .add_rule(C::Grass, D::Left, C::Grass | C::Dirt | C::Rock);
    wfc.rules
        .add_rule(C::Grass, D::Right, C::Grass | C::Dirt | C::Rock);
    wfc.rules
        .add_rule(C::Grass, D::Up, C::Grass | C::Dirt | C::Rock);
    wfc.rules
        .add_rule(C::Grass, D::Down, C::Grass | C::Dirt | C::Rock);

    wfc.rules.add_rule(C::Water, D::Left, C::Water | C::Grass);
    wfc.rules.add_rule(C::Water, D::Right, C::Water | C::Grass);
    wfc.rules.add_rule(C::Water, D::Up, C::Water | C::Grass);
    wfc.rules.add_rule(C::Water, D::Down, C::Water | C::Grass);

    wfc.rules
        .add_rule(C::Dirt, D::Left, C::Grass | C::Rock | C::Dirt);
    wfc.rules
        .add_rule(C::Dirt, D::Right, C::Grass | C::Rock | C::Dirt);
    wfc.rules
        .add_rule(C::Dirt, D::Up, C::Grass | C::Rock | C::Dirt);
    wfc.rules
        .add_rule(C::Dirt, D::Down, C::Grass | C::Rock | C::Dirt);

    wfc.rules
        .add_rule(C::Rock, D::Left, C::Rock | C::Dirt | C::Grass);
    wfc.rules
        .add_rule(C::Rock, D::Right, C::Rock | C::Dirt | C::Grass);
    wfc.rules
        .add_rule(C::Rock, D::Up, C::Rock | C::Dirt | C::Grass);
    wfc.rules
        .add_rule(C::Rock, D::Down, C::Rock | C::Dirt | C::Grass);

    //wfc.rules.add_rule(C::Floor, D::Left, C::Wall | C::Floor);
    //wfc.rules.add_rule(C::Floor, D::Right, C::Wall | C::Floor);
    //wfc.rules.add_rule(C::Floor, D::Up, C::Wall | C::Floor);
    //wfc.rules.add_rule(C::Floor, D::Down, C::Wall | C::Floor);

    //wfc.rules
    //.add_rule(C::Wall, D::Left, C::Wall | C::Grass | C::Floor);
    //wfc.rules
    //.add_rule(C::Wall, D::Right, C::Wall | C::Grass | C::Floor);
    //wfc.rules
    //.add_rule(C::Wall, D::Up, C::Wall | C::Grass | C::Floor);
    //wfc.rules
    //.add_rule(C::Wall, D::Down, C::Wall | C::Grass | C::Floor);

    wfc.weights.add_weight(C::Grass, C::Grass, 5.);
    wfc.weights.add_weight(C::Grass, C::Dirt, 2.);
    wfc.weights.add_weight(C::Dirt, C::Dirt, 0.1);
    wfc.weights.add_weight(C::Grass, C::Rock, 0.05);
    wfc.weights.add_weight(C::Rock, C::Rock, 3.);
}

fn reset(
    mut wfc: ResMut<Wfc>,
    keyboard_input: Res<Input<KeyCode>>,
    tiles: Query<Entity, With<Tile>>,
    mut commands: Commands,
) {
    if keyboard_input.just_pressed(KeyCode::R) {
        wfc.init();
        for entity in tiles.iter() {
            commands.entity(entity).despawn_recursive();
        }
    }
}

fn pause_time(mut time: ResMut<Time<Virtual>>) {
    time.pause();
}

fn update_mouse(
    camera_query: Query<(&Camera, &GlobalTransform)>,
    window_query: Query<&Window>,
    buttons: Res<Input<MouseButton>>,
    mut wfc: ResMut<Wfc>,
    mut force_reload: ResMut<ForceReload>,
) {
    let (camera, camera_transform) = camera_query.single();

    force_reload.0 = None;

    if let Some(cursor_position) = window_query.single().cursor_position() {
        if let Some(point) = camera.viewport_to_world_2d(camera_transform, cursor_position) {
            let coords = UVec2::new(
                (point.x / TILESIZE).floor() as u32,
                (point.y / TILESIZE).floor() as u32,
            );
            let mut v = wfc.get(coords).bits();
            let mut c = 0;
            while v > 1 {
                v >>= 1;
                c += 1;
            }
            let v = 1 << c;
            if buttons.just_pressed(MouseButton::Left) {
                wfc.set(coords, TileContent::from_bits_truncate(v << 1));
                force_reload.0 = Some(coords);
            }
            if buttons.just_pressed(MouseButton::Right) {
                wfc.set(coords, TileContent::from_bits_truncate(v >> 1));
                force_reload.0 = Some(coords);
            }
        };
    };
}
