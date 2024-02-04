use image::{ColorType, GenericImageView};
use rand::{rngs, seq::SliceRandom, Rng, SeedableRng};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    env,
};

struct Wfc {
    /// input image width
    width: usize,
    /// input image height
    height: usize,
    /// input image format
    format: image::ColorType,
    /// bytes per pixel
    bpp: usize,

    /// input data
    sampled_data: Vec<u32>,

    /// the data we work on
    data: Vec<Vec<u32>>,

    /// used to count intermediate images
    img_count: usize,

    /// how many pixels to look around to infer rules
    range: usize,
    /// the list of allowed states
    states: Vec<u32>,
    /// the inferred rules and weights
    /// ```
    /// rules[state][direction][state] = weight
    /// ```
    /// careful, rules[state] is empty for direction = center
    rules: HashMap<u32, Vec<HashMap<u32, f32>>>,
}

impl Wfc {
    /// Create a new Wfc from an image
    /// * `range` - the range to look around to infer rules
    /// * `img` - the image to ingest
    fn from_image(range: usize, img: &image::DynamicImage) -> Self {
        let (width, height) = img.dimensions();
        println!("ingested {}x{} {:?}", width, height, img.color());
        let mut wfc = Self {
            width: 0,
            height: 0,
            data: Vec::new(),
            sampled_data: Vec::new(),
            states: Vec::new(),
            range,
            rules: HashMap::new(),
            format: img.color(),
            bpp: 0,
            img_count: 0,
        };

        wfc.parse_data(&img.as_bytes().to_vec(), width as usize, height as usize);
        wfc.infer_rules(width as usize, height as usize);

        wfc
    }

    fn parse_data(&mut self, data: &Vec<u8>, width: usize, height: usize) {
        let mut states: HashSet<u32> = HashSet::new();

        self.sampled_data = Vec::new();
        self.sampled_data.resize((width * height) as usize, 0);

        self.bpp = match self.format {
            ColorType::Rgb8 => 3,
            ColorType::Rgba8 => 4,
            _ => panic!("unsupported format {:?}", self.format),
        };

        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * self.bpp) as usize;
                let r = data[idx];
                let g = data[idx + 1];
                let b = data[idx + 2];
                let v = match self.format {
                    ColorType::Rgb8 => (r as u32) << 16 | (g as u32) << 8 | (b as u32),
                    ColorType::Rgba8 => {
                        let a = data[idx + 3];
                        (r as u32) << 24 | (g as u32) << 16 | (b as u32) << 8 | a as u32
                    }
                    _ => {
                        panic!("unsupported format {:?}", self.format);
                    }
                };
                self.sampled_data[(y * width + x) as usize] = v;
                states.insert(v);
            }
        }

        println!("found {} states", states.len());
        for state in states {
            println!("\tstate: 0x{:08x}", state);
            self.states.push(state);
        }
    }

    fn infer_rules(&mut self, width: usize, height: usize) {
        //                 state  direction states
        //                     \     |     /
        let mut rules: HashMap<u32, Vec<Vec<u32>>> = HashMap::with_capacity(self.states.len());

        // width (and height) of the directions grid
        let dirs_width = self.range * 2 + 1;
        self.states.iter().for_each(|&state| {
            rules.insert(state, vec![Vec::new(); dirs_width.pow(2)]);
        });

        for x in 0..width {
            for y in 0..height {
                let v = self.sampled_data[(y * width + x) as usize];
                for subrange_x in 0..dirs_width {
                    for subrange_y in 0..dirs_width {
                        // do not use the cell were working on for the rules
                        if subrange_x == self.range && subrange_y == self.range {
                            continue;
                        }
                        if (x + subrange_x).checked_sub(self.range).is_none() {
                            continue;
                        }
                        if (y + subrange_y).checked_sub(self.range).is_none() {
                            continue;
                        }
                        if x + subrange_x - self.range >= width {
                            continue;
                        }
                        if y + subrange_y - self.range >= height {
                            continue;
                        }

                        let v1 = self.sampled_data[((x + subrange_x - self.range)
                            + (y + subrange_y - self.range) * width)
                            as usize];

                        rules.get_mut(&v).unwrap()[subrange_y * dirs_width + subrange_x].push(v1);
                    }
                }
            }
        }

        for (state, directions) in rules.iter() {
            let directions = directions
                .iter()
                .map(|states| {
                    let mut flattened_weights = HashMap::new();
                    let mut weights = HashMap::new();
                    let mut sum = 0;

                    states.iter().for_each(|state| {
                        *flattened_weights.entry(*state).or_insert(0) += 1;
                        sum += 1;
                    });

                    for (state, count) in flattened_weights.iter() {
                        weights.insert(*state, *count as f32 / sum as f32);
                    }
                    weights
                })
                .collect();
            self.rules.insert(*state, directions);
        }
    }

    fn debug_rules(&self) {
        let mut state_name = HashMap::new();
        state_name.insert(0x00000000, "black");
        state_name.insert(0x00ffffff, "white");
        println!();
        println!();
        println!();
        println!();
        println!("rules:");
        for (state, directions) in self.rules.iter() {
            if let Some(name) = state_name.get(state) {
                println!("state: {}", name);
            } else {
                println!("state: 0x{:08x}", state);
            }
            for (i, states) in directions.iter().enumerate() {
                println!("  dir: {}", i);
                for (state, weight) in states.iter() {
                    if let Some(name) = state_name.get(state) {
                        println!("    {} {:.2}", name, weight);
                    } else {
                        println!("    0x{:08x} {:.2}", state, weight);
                    }
                }
            }
        }
    }

    fn write_intermediary_image(&mut self) {
        let buffer = &self
            .data
            .iter()
            .flat_map(|s| {
                let bytes = s.iter().map(|v| v.to_be_bytes()).collect::<Vec<[u8; 4]>>();
                let mut mean_per_byte = vec![0; self.bpp];
                for (i, byte) in bytes.iter().enumerate() {
                    for j in 0..self.bpp {
                        mean_per_byte[j] = (mean_per_byte[j] as i32
                            + (byte[(4 - self.bpp) + j] as i32 - mean_per_byte[j] as i32)
                                / (i as i32 + 1)) as u8;
                    }
                }
                mean_per_byte
            })
            .collect::<Vec<u8>>();
        // ffmpeg -y -framerate 100 -i './out/img_%05d.bmp' -vf 'scale=1024:1024:flags=neighbor' video.mp4
        image::save_buffer(
            format!("out/img_{:05}.bmp", self.img_count).as_str(),
            buffer,
            self.width as u32,
            self.height as u32,
            self.format,
        )
        .unwrap();
        self.img_count += 1;
    }

    //                                                                   states    idx    dir (from neighbor's pov)
    fn get_neighbours(&self, x: usize, y: usize, range: usize) -> Vec<(&Vec<u32>, usize, usize)> {
        let mut neighbours = Vec::new();

        for ii in 0..(range * 2 + 1).pow(2) {
            let xx = ii % (range * 2 + 1);
            let yy = ii / (range * 2 + 1);
            if (x + xx).checked_sub(range).is_some()
                && (y + yy).checked_sub(range).is_some()
                && x + xx < self.width
                && y + yy < self.height
            {
                let i = ((x + xx - range) + (y + yy - range) * self.width) as usize;
                let neighbor_states = &self.data[i];
                let (rx, ry) = (2 * range - xx, 2 * range - yy);
                neighbours.push((neighbor_states, i, ry * (self.range * 2 + 1) + rx));
            }
        }

        neighbours
    }

    // TODO
    // maybe we should store this data in the tile data ? here we recompute it a LOT
    fn get_weighted_possible_states(&self, x: usize, y: usize) -> Vec<(u32, f32)> {
        let states = self.data[x + y * self.width].clone();
        let mut weighted_states = Vec::new();
        for state in states {
            let mut weight = 1.;
            for (neighbor_states, _, dir) in self.get_neighbours(x, y, self.range) {
                if neighbor_states.len() == 1 {
                    if let Some(w) = self.rules[&neighbor_states[0]][dir].get(&state) {
                        weight += w;
                    } else {
                        weight = 0.;
                        break;
                    }
                }
            }
            weighted_states.push((state, weight));
        }
        weighted_states
    }

    fn gen(&mut self, width: usize, height: usize, seed: Option<u64>) -> Vec<u32> {
        self.width = width;
        self.height = height;

        let seed = seed.unwrap_or_else(|| rand::thread_rng().gen());
        let mut rng = rngs::StdRng::seed_from_u64(seed);
        println!("seed {}", seed);
        self.data
            .resize((width * height) as usize, self.states.clone());
        let mut propagation: VecDeque<usize> = VecDeque::new();
        let mut observed = 0;
        let mut time = std::time::Instant::now();

        loop {
            if time.elapsed().as_secs() >= 1 {
                println!("{:.2}%", observed as f32 / (width * height) as f32 * 100.);
                time = std::time::Instant::now();
            }
            if let Some(to_collapse) = propagation.pop_front() {
                let x = to_collapse % width;
                let y = to_collapse / width;
                let states = self.data[x + y * width].clone();

                if states.len() <= 1 {
                    continue;
                }

                // get allowed states
                let allowed_states: Vec<u32> = states
                    .iter()
                    .filter_map(|&state| {
                        // for each neighbour
                        // TODO check only nearest neighbours (range = 1 instead of self.range) ?
                        // TODO ? or not ? maybe we should try all its possible states if its not collapsed
                        for (neighbor_states, _, dir) in self.get_neighbours(x, y, self.range) {
                            if neighbor_states.len() == 1 {
                                if !self.rules[&neighbor_states[0]][dir].contains_key(&state) {
                                    // the state is not allowed
                                    return None;
                                }
                            }
                        }
                        // the state is allowed, because no neighbour returned None before
                        Some(state)
                    })
                    .collect();
                if allowed_states.len() == 1 {
                    observed += 1;
                }
                if allowed_states.len() != self.data[x + y * width].len() {
                    self.data[x + y * width] = allowed_states;
                    // TODO same, maybe we can just push the range 1 neighbours ?
                    for (_, index, _) in self.get_neighbours(x, y, self.range) {
                        propagation.push_back(index);
                    }
                }
            } else {
                self.write_intermediary_image();
                let tiles_with_entropy: Vec<(usize, f32)> = self
                    .data
                    .iter()
                    .enumerate()
                    .filter_map(|(i, states)| {
                        if states.len() > 1 {
                            let x = i % self.width;
                            let y = i / self.width;
                            Some((
                                i,
                                -self.get_weighted_possible_states(x, y).iter().fold(
                                    0.,
                                    // shannon's entropy (if not, all the tiles have the same entropy, especially on simple samples)
                                    |acc, (_, w)| {
                                        // log2(0) is -inf, and 0 * -inf is nan
                                        // so we just cut to -inf
                                        if *w > 0. {
                                            acc + w * w.log2()
                                        } else {
                                            acc + -f32::INFINITY
                                        }
                                    },
                                ),
                            ))
                        } else {
                            None
                        }
                    })
                    .collect();

                if tiles_with_entropy.is_empty() {
                    println!("done");
                    return self.data.iter().map(|s| *s.get(0).unwrap_or(&0)).collect();
                }
                let (_, lowest_entropy) = tiles_with_entropy
                    .iter()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();
                let i = tiles_with_entropy
                    .iter()
                    .filter(|(_, entropy)| *entropy == *lowest_entropy)
                    .collect::<Vec<_>>()
                    .choose(&mut rng)
                    .unwrap()
                    .0;

                let x = i % width;
                let y = i / width;
                let weighted_states = self.get_weighted_possible_states(x, y);
                if let Ok(state) = weighted_states.choose_weighted(&mut rng, |(_, weight)| *weight)
                {
                    self.data[i] = vec![state.0];
                } else {
                    self.data[i] = vec![];
                }
                observed += 1;
                for (_, index, _) in self.get_neighbours(x, y, self.range) {
                    propagation.push_back(index);
                }
            }
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let Some(filename) = args.get(1) else {
        panic!("usage: {} ./path_to_image.bmp", args[0]);
    };

    let Ok(img) = image::open(filename) else {
        panic!("failed to open {}", filename);
    };

    let mut wfc = Wfc::from_image(1, &img);
    //wfc.debug_rules();
    let (w, h) = (64 as u32, 64 as u32);
    let start = std::time::Instant::now();
    let data = wfc.gen(w as usize, h as usize, None);
    println!("generated in {:?}", start.elapsed());

    let buffer: Vec<u8> = data
        .iter()
        .flat_map(|v| {
            v.to_be_bytes()
                .iter()
                .cloned()
                .skip(4 - wfc.bpp)
                .take(wfc.bpp)
                .collect::<Vec<_>>()
        })
        .collect();
    image::save_buffer("out.bmp", &buffer, w, h, wfc.format).unwrap()
}
