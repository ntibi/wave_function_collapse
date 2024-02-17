use image::{ColorType, GenericImageView};
use rand::{rngs, seq::SliceRandom, Rng, SeedableRng};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    env,
    num::Wrapping,
    ops::{Index, IndexMut},
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

struct WfcGenerator {
    range: usize,
    patterns: Vec<Pattern>,
    // rules[pattern_id][dir][pattern_id] = weight
    rules: Vec<Vec<Vec<f32>>>,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
struct PatternId(usize);

impl<T> Index<PatternId> for Vec<T> {
    type Output = T;

    fn index(&self, i: PatternId) -> &T {
        &self[i.0]
    }
}

impl<T> IndexMut<PatternId> for Vec<T> {
    fn index_mut(&mut self, i: PatternId) -> &mut T {
        &mut self[i.0]
    }
}

impl WfcGenerator {
    /// Create a new WfcGenerator from an image
    /// * `range` - the range to look around to infer rules
    /// * `img` - the image to ingest
    fn from_image(range: usize, img: &image::DynamicImage) -> Self {
        let (width, height) = img.dimensions();

        let data = WfcGenerator::parse_data(&img, width as usize, height as usize);
        let (patterns, rules) =
            WfcGenerator::infer_patterns(&data, range, width as usize, height as usize);
        println!("patterns: {:?}", patterns);
        println!("rules: {:?}", rules);

        WfcGenerator {
            range,
            patterns,
            rules,
        }
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
    ) -> (Vec<Pattern>, Vec<Vec<Vec<f32>>>) {
        let mut patterns: HashMap<Pattern, PatternId> = HashMap::new();
        let mut rules: Vec<Vec<Vec<PatternId>>> = Vec::new();
        let mut pattern_id = PatternId(0);
        let mut mapped: Vec<Option<PatternId>> = vec![None; data.len()];

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

                let i = (x + y * width) as usize;

                if let Some(pattern_id) = patterns.get_mut(&Pattern::new(pattern.clone())) {
                    mapped[i] = Some(*pattern_id);
                } else {
                    patterns.insert(Pattern::new(pattern.clone()), pattern_id);
                    mapped[i] = Some(pattern_id);
                    pattern_id.0 += 1;
                }
            }
        }

        let n_patterns = pattern_id.0;
        println!("found {} patterns", n_patterns);

        rules = vec![vec![Vec::new(); (range * 2 + 1).pow(2)]; n_patterns];

        for x in range..width - range {
            for y in range..height - range {
                let pattern_id = mapped[(x + y * width) as usize].unwrap();
                for xx in 0..(range * 2 + 1) {
                    for yy in 0..(range * 2 + 1) {
                        let ii = (x + xx - range) + (y + yy - range) * width;
                        let x = Wrapping(range);
                        let y = Wrapping(range);
                        let xx = Wrapping(xx);
                        let yy = Wrapping(yy);
                        let size = Wrapping(range * 2 + 1);
                        let half_pow = Wrapping((range * 2 + 1).pow(2) / 2);
                        let dir = (xx - x + (yy - y) * (size) + (half_pow)).0;
                        if let Some(mapped) = mapped[ii] {
                            rules[pattern_id][dir].push(mapped);
                        }
                    }
                }
            }
        }

        let patterns = patterns
            .into_iter()
            .map(|(pattern, _)| pattern)
            .collect::<Vec<_>>();

        let rules = rules
            .iter()
            .map(|pid| {
                pid.iter()
                    .map(|dir| {
                        let mut out = vec![0.0_f32; n_patterns];
                        let mut total = 0.;
                        for p in dir.iter() {
                            out[p.0] += 1.;
                            total += 1.;
                        }
                        for p in out.iter_mut() {
                            *p /= total;
                        }
                        out
                    })
                    .collect()
            })
            .collect();

        (patterns, rules)
    }

    /// do these pattern match in the given direction
    fn check_overlap(&self, pattern: &Pattern, dir: usize, other: &Pattern) -> bool {
        let other_relative_dir = (self.range * 2 + 1).pow(2) - dir - 1;
        pattern.pattern[dir] == other.pattern[other_relative_dir]
    }

    fn get_pattern(&self, pattern_id: PatternId) -> &Pattern {
        &self.patterns[pattern_id]
    }

    fn get_pattern_weight(
        &self,
        pattern: PatternId,
        data: &Vec<Vec<(PatternId, f32)>>,
        x: usize,
        y: usize,
        width: usize,
        height: usize,
    ) -> f32 {
        let neighbours = self.get_neighbours(x, y, width, height);
        let mut weight = 0.;

        // for each neighbour of the input cell
        for neighbour in neighbours.iter() {
            let neighbour_patterns = &data[*neighbour];
            // if the neighbour is not collapsed, we skip it
            if neighbour_patterns.len() != 1 {
                continue;
            }
            let neighbour_x = *neighbour % width;
            let neighbour_y = *neighbour / width;

            let x = Wrapping(x);
            let y = Wrapping(y);
            let neighbour_x = Wrapping(neighbour_x);
            let neighbour_y = Wrapping(neighbour_y);
            let size = Wrapping(self.range * 2 + 1);
            let half_pow = Wrapping((self.range * 2 + 1).pow(2) / 2);
            let dir = (neighbour_x - x + (neighbour_y - y) * (size) + (half_pow)).0;
            let w = self.rules[pattern][dir][neighbour_patterns[0].0];
            if w == 0. {
                return 0.;
            } else {
                weight += w;
            }
        }

        weight
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
        let mut weights: Vec<f32> = vec![0.; self.patterns.len()];
        for (pattern_id, _) in data[y * width + x].iter() {
            let w = self.get_pattern_weight(*pattern_id, data, x, y, width, height);
            weights[*pattern_id] += w;
        }

        let weights: Vec<(PatternId, f32)> = weights
            .iter()
            .enumerate()
            .filter_map(|(i, w)| {
                if *w > 0. {
                    Some((PatternId(i), *w))
                } else {
                    None
                }
            })
            .collect();
        let sum = weights.iter().fold(0., |acc, (_, w)| acc + w);
        weights.iter().map(|(i, w)| (*i, *w / sum)).collect()
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
                // TODO use input weight distribution instead of even one ?
                (pid, 1. / self.patterns.len() as f32)
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
                            Some(v) => self.get_pattern(v.0).get_center(),
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

    let wfc = WfcGenerator::from_image(1, &img);
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
