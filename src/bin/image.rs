use image::{ColorType, GenericImageView};
use rand::{rngs, seq::SliceRandom, Rng, SeedableRng};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    env,
    num::Wrapping,
};

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct Pattern {
    pattern: Vec<u32>,
}

impl Pattern {
    fn new(pattern: Vec<u32>) -> Self {
        Pattern { pattern }
    }

    fn get_center(&self) -> u32 {
        self.pattern[self.pattern.len() / 2]
    }
}

#[derive(Clone, Debug)]
struct WeightedPattern {
    pattern: Pattern,
    weight: f32,
}

impl WeightedPattern {
    fn new(pattern: Pattern, weight: f32) -> Self {
        WeightedPattern { pattern, weight }
    }
}

struct WfcGenerator {
    range: usize,
    patterns: Vec<WeightedPattern>,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
struct PatternId(usize);

impl WfcGenerator {
    /// Create a new WfcGenerator from an image
    /// * `range` - the range to look around to infer rules
    /// * `img` - the image to ingest
    fn from_image(range: usize, img: &image::DynamicImage) -> Self {
        let (width, height) = img.dimensions();

        let data = WfcGenerator::parse_data(&img, width as usize, height as usize);
        let patterns = WfcGenerator::infer_patterns(&data, range, width as usize, height as usize);

        WfcGenerator { patterns, range }
    }

    fn parse_data(img: &image::DynamicImage, width: usize, height: usize) -> Vec<u32> {
        let data: Vec<u8> = img.as_bytes().to_vec();
        let mut out: Vec<u32> = vec![0; width * height];

        let format = img.color();
        let bpp = match format {
            ColorType::Rgb8 => 3,
            ColorType::Rgba8 => 4,
            _ => panic!("unsupported format {:?}", format),
        };

        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * bpp) as usize;
                let r = data[idx];
                let g = data[idx + 1];
                let b = data[idx + 2];
                let v = match format {
                    ColorType::Rgb8 => (r as u32) << 16 | (g as u32) << 8 | (b as u32),
                    ColorType::Rgba8 => {
                        let a = data[idx + 3];
                        (r as u32) << 24 | (g as u32) << 16 | (b as u32) << 8 | a as u32
                    }
                    _ => {
                        panic!("unsupported format {:?}", format);
                    }
                };
                out[(y * width + x) as usize] = v;
            }
        }

        out
    }

    fn infer_patterns(
        data: &Vec<u32>,
        range: usize,
        width: usize,
        height: usize,
    ) -> Vec<WeightedPattern> {
        let mut patterns: HashMap<Pattern, usize> = HashMap::new();

        let pattern_size = range * 2 + 1;

        let permutations_fns: Vec<Box<dyn Fn(usize, usize) -> (usize, usize)>> = vec![
            // None
            Box::new(|x: usize, y: usize| (x, y)),
            // Y axis symmetry
            Box::new(move |x: usize, y: usize| (pattern_size - 1 - x, y)),
            // X axis symmetry
            Box::new(move |x: usize, y: usize| (x, pattern_size - 1 - y)),
            // XY symmetry
            Box::new(move |x: usize, y: usize| (pattern_size - 1 - x, pattern_size - 1 - y)),
            // 90° clockwise
            Box::new(move |x: usize, y: usize| (y, pattern_size - 1 - x)),
            // 90° counter clockwise
            Box::new(move |x: usize, y: usize| (pattern_size - 1 - y, x)),
            // Y axis symmetry + 90° clockwise
            Box::new(move |x: usize, y: usize| (pattern_size - 1 - y, pattern_size - 1 - x)),
            // Y axis symmetry + 90° clockwise
            Box::new(move |x: usize, y: usize| (y, x)),
        ];

        for x in range..width - range {
            for y in range..height - range {
                let mut pattern = vec![0; (pattern_size).pow(2)];
                for subrange_x in 0..pattern_size {
                    for subrange_y in 0..pattern_size {
                        //for permutation_fn in permutations_fns.iter() {
                        //let (subrange_x, subrange_y) = permutation_fn(subrange_x, subrange_y);

                        let i = subrange_x + subrange_y * pattern_size;

                        pattern[i] = data[((x + subrange_x - range)
                            + (y + subrange_y - range) * width)
                            as usize];

                        //}
                    }
                }
                patterns
                    .entry(Pattern::new(pattern.clone()))
                    .and_modify(|entry| *entry += 1)
                    .or_insert(1);
            }
        }

        println!("found {} patterns", patterns.len());

        let total = patterns.values().sum::<usize>() as f32;
        patterns
            .iter()
            .map(|(p, &w)| WeightedPattern::new(p.clone(), w as f32 / total))
            .collect()
    }

    /// do these pattern match in the given direction
    fn check_overlap(&self, pattern: &Pattern, dir: usize, other: &Pattern) -> bool {
        let other_relative_dir = (self.range * 2 + 1).pow(2) - dir - 1;
        pattern.pattern[dir] == other.pattern[other_relative_dir]
    }

    fn get_pattern(&self, pattern_id: PatternId) -> &WeightedPattern {
        &self.patterns[pattern_id.0]
    }

    fn is_allowed(
        &self,
        pattern: Pattern,
        data: &Vec<Vec<(PatternId, f32)>>,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> bool {
        let neighbours = self.get_neighbours(x, y, width, height);
        // for each neighbour of the input cell
        for neighbour in neighbours.iter() {
            let neighbour_patterns = &data[*neighbour];
            // if the neighbour is not collapsed, we skip it
            if neighbour_patterns.len() != 1 {
                continue;
            }

            // TODO we could have a lookup table where we precompute
            // all the possible patterns, for each pattern, for each direction
            let neighbour_pattern = self.get_pattern(neighbour_patterns[0].0);
            for neighbour_x in 0..(self.range * 2 + 1) {
                for neighbour_y in 0..(self.range * 2 + 1) {
                    let neighbour_index = neighbour_x + neighbour_y * (self.range * 2 + 1);
                    // if the cell around the neighbour is not in the range of the input cell, we skip it
                    // TODO we could do this just by modifying the double x,y loop, but im too lazy for now
                    //   rn this is extremely inefficient
                    if !neighbours.contains(&neighbour_index) {
                        continue;
                    }
                    // dir relative to the input cell
                    let x = Wrapping(x);
                    let y = Wrapping(y);
                    let neighbour_x = Wrapping(neighbour_x);
                    let neighbour_y = Wrapping(neighbour_y);
                    let size = Wrapping(self.range * 2 + 1);
                    let half_pow = Wrapping((self.range * 2 + 1).pow(2) / 2);
                    let dir = (neighbour_x - x + (neighbour_y - y) * (size) + (half_pow)).0;
                    if !self.check_overlap(&pattern, dir, &neighbour_pattern.pattern) {
                        return false;
                    }
                }
            }
        }
        true
    }

    fn get_weighted_possible_patterns(
        &self,
        data: &Vec<Vec<(PatternId, f32)>>,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> Vec<(PatternId, f32)> {
        // for each pattern id
        let mut possible_patterns: Vec<(PatternId, f32)> = Vec::new();
        for (pattern_id, _) in data[y * width + x].iter() {
            let pattern = self.get_pattern(*pattern_id);
            if self.is_allowed(pattern.pattern.clone(), data, x, y, width, height) {
                // TODO weighting
                possible_patterns.push((*pattern_id, 1.));
            }
        }

        possible_patterns
    }

    fn get_neighbours(&self, x: usize, y: usize, width: usize, height: usize) -> Vec<usize> {
        let mut neighbours = Vec::new();

        for dir in 0..(self.range * 2 + 1).pow(2) {
            // do not return self
            if dir == self.range * (self.range * 2 + 1) + self.range {
                continue;
            }
            let xx = dir % (self.range * 2 + 1);
            let yy = dir / (self.range * 2 + 1);
            if (x + xx).checked_sub(self.range).is_some()
                && (y + yy).checked_sub(self.range).is_some()
                && x + xx <= width
                && y + yy <= height
            {
                let i = ((x + xx - self.range) + (y + yy - self.range) * width) as usize;
                neighbours.push(i);
            }
        }

        neighbours
    }

    fn gen(&self, width: usize, height: usize, seed: Option<u64>) -> Vec<u32> {
        let seed = seed.unwrap_or_else(|| rand::thread_rng().gen());
        println!("seed: {}", seed);
        let mut rng = rngs::StdRng::seed_from_u64(seed);

        let weighted_pattern_ids: Vec<(PatternId, f32)> = (0..self.patterns.len())
            .map(|id| {
                let pid = PatternId(id);
                (pid, self.get_pattern(pid).weight)
            })
            .collect::<Vec<_>>();
        let mut data = vec![weighted_pattern_ids.clone(); width * height];
        let mut stack: VecDeque<usize> = VecDeque::new();

        loop {
            if let Some(to_observe) = stack.pop_front() {
                let patterns = &data[to_observe];
                // we are exploring a collapsed cell
                if patterns.len() == 1 {
                    continue;
                }
                let (x, y) = (to_observe % width, to_observe / width);

                let new_patterns = self.get_weighted_possible_patterns(&data, x, y, width, height);
                // if we had a change, we apply it and propagate
                if new_patterns != *patterns {
                    data[to_observe] = new_patterns;
                    for neighbour in self.get_neighbours(x, y, width, height).iter() {
                        // only push non collapsed neighbours
                        if data[*neighbour].len() > 1 {
                            stack.push_back(*neighbour);
                        }
                    }
                }
            } else {
                let indexes_with_entropy: Vec<(usize, f32)> = data
                    .iter()
                    .enumerate()
                    .filter_map(|(i, weighted_patterns)| {
                        if weighted_patterns.len() > 1 {
                            Some((
                                i,
                                -weighted_patterns.iter().fold(
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

                // if we have no entropy, we are done
                if indexes_with_entropy.is_empty() {
                    return data
                        .iter()
                        .map(|s| match s.get(0) {
                            Some(v) => self.get_pattern(v.0).pattern.get_center(),
                            None => 0,
                        })
                        .collect::<Vec<u32>>();
                }

                let (_, lowest_entropy) = indexes_with_entropy
                    .iter()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();
                let i = indexes_with_entropy
                    .iter()
                    .filter(|(_, entropy)| *entropy == *lowest_entropy)
                    .collect::<Vec<_>>()
                    .choose(&mut rng)
                    .unwrap()
                    .0;

                match data[i].choose_weighted(&mut rng, |(_, weight)| *weight) {
                    Ok((pattern, _)) => {
                        data[i] = vec![(*pattern, 1.)];
                    }
                    Err(e) => {
                        println!("failed to choose a pattern {}", e);
                        data[i] = vec![];
                    }
                }

                let (x, y) = (i % width, i / width);
                for neighbour in self.get_neighbours(x, y, width, height).iter() {
                    // only push non collapsed neighbours
                    if data[*neighbour].len() > 1 {
                        stack.push_back(*neighbour);
                    }
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

    let mut wfc = WfcGenerator::from_image(1, &img);
    let (w, h) = (16 as u32, 16 as u32);
    let start = std::time::Instant::now();
    let data = wfc.gen(w as usize, h as usize, None);
    println!("generated in {:?}", start.elapsed());

    let bpp = match img.color() {
        ColorType::Rgb8 => 3,
        ColorType::Rgba8 => 4,
        _ => panic!("unsupported format {:?}", img.color()),
    };
    let buffer: Vec<u8> = data
        .iter()
        .flat_map(|v| {
            v.to_be_bytes()
                .iter()
                .cloned()
                .skip(4 - bpp)
                .take(bpp)
                .collect::<Vec<_>>()
        })
        .collect();
    image::save_buffer("out.bmp", &buffer, w, h, img.color()).unwrap()
}
