use image::{ColorType, GenericImageView};
use rand::{rngs, seq::SliceRandom, Rng, SeedableRng};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    env,
};

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct Pattern {
    pattern: Vec<u32>,
}

impl Pattern {
    fn new(pattern: Vec<u32>) -> Self {
        Pattern { pattern }
    }
}

#[derive(Debug)]
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

impl WfcGenerator {
    /// Create a new WfcGenerator from an image
    /// * `range` - the range to look around to infer rules
    /// * `img` - the image to ingest
    fn from_image(range: usize, img: &image::DynamicImage) -> Self {
        let (width, height) = img.dimensions();

        let data = WfcGenerator::parse_data(&img, width as usize, height as usize);
        let patterns = WfcGenerator::infer_patterns(&data, range, width as usize, height as usize);
        println!("patterns: {:?}", patterns);

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

        patterns
            .iter()
            .map(|(p, &w)| WeightedPattern::new(p.clone(), w as f32))
            .collect()
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
    //wfc.debug_rules();
    //let (w, h) = (3 as u32, 3 as u32);
    //let start = std::time::Instant::now();
    //let data = wfc.gen(w as usize, h as usize, None);
    ////let data = wfc.gen(w as usize, h as usize, Some(17550241477713680911));
    //println!("generated in {:?}", start.elapsed());

    //let buffer: Vec<u8> = data
    //.iter()
    //.flat_map(|v| {
    //v.to_be_bytes()
    //.iter()
    //.cloned()
    //.skip(4 - wfc.bpp)
    //.take(wfc.bpp)
    //.collect::<Vec<_>>()
    //})
    //.collect();
    //image::save_buffer("out.bmp", &buffer, w, h, wfc.format).unwrap()
}
