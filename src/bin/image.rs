use image::{ColorType, GenericImageView, ImageBuffer};
use rand::{rngs, seq::SliceRandom, thread_rng, Rng, SeedableRng};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    env,
};

/// this iterator will yield all the coords of a square around a point (except for the point itself)
/// ```
/// Iter2D::new(1).for_each(|(x, y)| println!("{},{}", x, y));
/// > 0,0
/// > 0,1
/// > 0,2
/// > 1,0
/// // notice the missing 1,1
/// > 1,2
/// > 2,0
/// > 2,1
/// > 2,2
/// ```
struct Iter2D {
    range: usize,
    current: (usize, usize),
}

impl Iter2D {
    fn new(range: usize) -> Self {
        Iter2D {
            current: (0, 0),
            range,
        }
    }
}

impl Iterator for Iter2D {
    type Item = (usize, usize);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.current {
                (x, y) if x == self.range && y == self.range => {
                    self.current.0 += 1;
                }
                (x, y) if x < self.range * 2 => {
                    self.current.0 += 1;
                    return Some((x, y));
                }
                (x, y) if y < self.range * 2 => {
                    self.current.0 = 0;
                    self.current.1 += 1;
                    return Some((x, y));
                }
                (x, y) if x == self.range * 2 && y == self.range * 2 => {
                    self.current.0 += 1;
                    return Some((x, y));
                }
                _ => return None,
            }
        }
    }
}

struct WfcInput {
    width: usize,
    height: usize,
    format: image::ColorType,

    /// input data
    data: Vec<u32>,

    /// how many pixels to look around to infer rules
    range: usize,
    /// the list of allowed states
    states: Vec<u32>,
    /// the inferred rules and weights
    /// rules[state][direction][state] = weight
    /// careful, rules[state][] is empty
    rules: HashMap<u32, Vec<HashMap<u32, f32>>>,
}

impl WfcInput {
    /// Create a new WfcInput from an image
    /// * `range` - the range to look around to infer rules
    /// * `img` - the image to ingest
    fn from_image(range: usize, img: &image::DynamicImage) -> Self {
        let (width, height) = img.dimensions();
        println!("ingested {}x{} {:?}", width, height, img.color());
        let mut b = Self {
            width: width as usize,
            height: height as usize,
            data: Vec::new(),
            states: Vec::new(),
            range,
            rules: HashMap::new(),
            format: img.color(),
        };

        b.parse_data(&img.as_bytes().to_vec());
        b.infer_rules();

        b
    }

    fn parse_data(&mut self, data: &Vec<u8>) {
        let mut states: HashSet<u32> = HashSet::new();

        self.data = Vec::new();
        self.data.resize((self.width * self.height) as usize, 0);

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = ((y * self.width + x) * 3) as usize;
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
                self.data[(y * self.width + x) as usize] = v;
                states.insert(v);
            }
        }

        println!("found {} states", states.len());
        for state in states {
            println!("state: 0x{:08x}", state);
            self.states.push(state);
        }
    }

    fn infer_rules(&mut self) {
        //                 state  direction states
        //                     \     |     /
        let mut rules: HashMap<u32, Vec<Vec<u32>>> = HashMap::with_capacity(self.states.len());

        // width (and height) of the directions grid
        let dirs_width = self.range * 2 + 1;
        self.states.iter().for_each(|&state| {
            rules.insert(state, vec![Vec::new(); dirs_width.pow(2)]);
        });

        for x in 0..self.width {
            for y in 0..self.height {
                let v = self.data[(y * self.width + x) as usize];
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
                        if x + subrange_x - self.range >= self.width {
                            continue;
                        }
                        if y + subrange_y - self.range >= self.height {
                            continue;
                        }

                        let v1 = self.data[((x + subrange_x - self.range)
                            + (y + subrange_y - self.range) * self.width)
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

    fn gen(&mut self, width: usize, height: usize, seed: Option<u64>) -> Vec<u32> {
        let seed = seed.unwrap_or_else(|| rand::thread_rng().gen());
        let mut rng = rngs::StdRng::seed_from_u64(seed);
        println!("seed {}", seed);
        let mut data = Vec::new();
        data.resize((width * height) as usize, self.states.clone());
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
                let states = data[x + y * width].clone();

                if states.len() <= 1 {
                    continue;
                }

                // get allowed states
                let allowed_states: Vec<u32> = states
                    .iter()
                    .filter_map(|&state| {
                        // for each neighbour
                        // TODO check only nearest neighbours (range = 1 instead of self.range) ?
                        for (xx, yy) in Iter2D::new(self.range) {
                            if (x + xx).checked_sub(self.range).is_some()
                                && (y + yy).checked_sub(self.range).is_some()
                                && x + xx < width
                                && y + yy < height
                            {
                                // get its value
                                let neighbor_states = &data[((x + xx - self.range)
                                    + (y + yy - self.range) * width)
                                    as usize];
                                // if its already collapsed
                                // TODO ? or not ? maybe we should try all its possible states if its not collapsed
                                if neighbor_states.len() == 1 {
                                    // xx, yy are relative to the tile we are working on
                                    // so we need to invert them to get the direction (here we do a point symetry on the center)
                                    let (rx, ry) = (2 * self.range - xx, 2 * self.range - yy);
                                    if !self.rules[&neighbor_states[0]]
                                        [ry * (self.range * 2 + 1) + rx]
                                        .contains_key(&state)
                                    {
                                        // the state is not allowed
                                        return None;
                                    }
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
                data[x + y * width] = allowed_states;
            } else {
                if let Some(lowest_entropy) = data
                    .iter()
                    .filter_map(|states| {
                        if states.len() > 1 {
                            Some(states.len())
                        } else {
                            None
                        }
                    })
                    .min()
                {
                    let lowest_entropy_indices: Vec<usize> = data
                        .iter()
                        .enumerate()
                        .filter_map(|(i, states)| {
                            if states.len() == lowest_entropy {
                                Some(i)
                            } else {
                                None
                            }
                        })
                        .collect();
                    let i = *lowest_entropy_indices.choose(&mut rng).unwrap();
                    let x = i % width;
                    let y = i / width;
                    data[i] = vec![*data[i]
                        .choose_weighted(&mut rng, |state| {
                            Iter2D::new(self.range)
                                .map(|(xx, yy)| {
                                    if (x + xx).checked_sub(self.range).is_some()
                                        && (y + yy).checked_sub(self.range).is_some()
                                        && x + xx < width
                                        && y + yy < height
                                    {
                                        let neighbor = &data[((x + xx - self.range)
                                            + (y + yy - self.range) * width)
                                            as usize];
                                        if neighbor.len() == 1 {
                                            let (rx, ry) =
                                                (2 * self.range - xx, 2 * self.range - yy);
                                            return *self.rules[&neighbor[0]]
                                                [ry * (self.range * 2 + 1) + rx]
                                                .get(state)
                                                .unwrap_or(&0.);
                                        }
                                    }
                                    1.
                                })
                                .sum::<f32>()
                        })
                        .unwrap()];
                    observed += 1;
                } else {
                    println!("done");
                    return data.iter().map(|s| s[0]).collect();
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

    let mut wfc = WfcInput::from_image(1, &img);
    let (w, h) = (128 as u32, 128 as u32);
    let start = std::time::Instant::now();
    let data = wfc.gen(w as usize, h as usize, None);
    println!("generated in {:?}", start.elapsed());

    let bytes_per_pixel = match wfc.format {
        ColorType::Rgb8 => 3,
        ColorType::Rgba8 => 4,
        _ => panic!("unsupported format {:?}", wfc.format),
    };

    let buffer: Vec<u8> = data
        .iter()
        .flat_map(|v| {
            v.to_le_bytes()
                .iter()
                .cloned()
                .take(bytes_per_pixel)
                .collect::<Vec<_>>()
        })
        .collect();
    image::save_buffer(filename, &buffer, w, h, wfc.format).unwrap()
}
