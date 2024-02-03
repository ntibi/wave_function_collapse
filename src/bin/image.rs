use image::GenericImageView;
use std::{
    collections::{HashMap, HashSet},
    env,
};

struct WfcInput {
    width: usize,
    height: usize,
    data: Vec<u32>,

    /// how many pixels to look around to infer rules
    range: usize,
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
                let idx = ((y * self.width + x) * 4) as usize;
                let r = data[idx];
                let g = data[idx + 1];
                let b = data[idx + 2];
                let a = data[idx + 3];
                let v = (r as u32) << 24 | (g as u32) << 16 | (b as u32) << 8 | a as u32;
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
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let Some(filename) = args.get(1) else {
        panic!("usage: {} ./path_to_image.bmp", args[0]);
    };

    let Ok(img) = image::open(filename) else {
        panic!("failed to open {}", filename);
    };

    let wfc = WfcInput::from_image(1, &img);
}
