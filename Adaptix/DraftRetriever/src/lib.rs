// The code is adapted from https://github.com/FasterDecoding/REST;
// The code for retrival is adapted from https://github.com/Intsights/PySubstringSearch;
// The code for drafft buffer is adapted from https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/utils.py#L31-L124
use bincode;
use bstr::io::BufReadExt;
use byteorder::{ByteOrder, LittleEndian, ReadBytesExt, WriteBytesExt};
use cached::proc_macro::cached;
use mcts_sub::MCTS;
use memchr::memmem;
use parking_lot::Mutex;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;
use std::cmp;
use std::cmp::max;
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::fs::File;
use std::fs::OpenOptions;
use std::hash::Hash;
use std::io::Cursor;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::rc::Rc;
use std::str;
use std::sync::Arc;
use std::time::Instant;
mod mcts_sub;
#[pyclass]
struct Writer {
    file_path: String,
    array: HashMap<(i32, i32), HashMap<i32, i32>>,
    vocab_size: i32,
}

#[pymethods]
impl Writer {
    #[new]
    fn new(file_path: &str, vocab_size: Option<i32>) -> PyResult<Self> {
        let file_path = file_path.to_string();
        let vocab_size = vocab_size.unwrap_or(35000);
        let mut array = HashMap::new();
        Ok(Writer {
            file_path,
            array,
            vocab_size,
        })
    }

    fn add_entry(&mut self, py_text: &PyList) -> PyResult<()> {
        let mut text = Vec::new();
        for item in py_text.iter() {
            let num: i32 = item.extract()?;
            text.push(num);
        }
        // 3-gram
        if text.len() < 3 {
            return Ok(());
        }
        for i in 0..text.len() - 2 {
            let key = (text[i], text[i + 1]);
            let count = self.array.entry(key).or_insert(HashMap::new());
            *count.entry(text[i + 2]).or_insert(0) += 1;
        }

        Ok(())
    }

    fn finalize(&mut self) -> PyResult<()> {
        // normalize 3-gram, put it in data
        let mut data: HashMap<(i32, i32), HashMap<i32, f32>> = HashMap::new();
        for (key, value) in &self.array {
            let mut total = 0;
            for (_, count) in value {
                total += count;
            }
            let mut normalized = HashMap::new();
            for (k, v) in value {
                normalized.insert(*k, *v as f32 / total as f32);
            }
            data.insert(key.clone(), normalized);
        }

        let mut index_file = OpenOptions::new()
            .write(true)
            .create(true)
            .open(&self.file_path)?;
        let encoded: Vec<u8> = bincode::serialize(&data).unwrap();
        index_file.write_all(&encoded).unwrap();
        Ok(())
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        self.finalize().unwrap();
    }
}

#[pyclass]
struct Reader {
    array: HashMap<(i32, i32), HashMap<i32, f32>>,
}

#[pymethods]
impl Reader {
    #[new]
    fn new(index_file_path: &str) -> PyResult<Self> {
        let mut index_file = File::open(index_file_path)?;
        let mut encoded = Vec::new();
        index_file.read_to_end(&mut encoded).unwrap();
        let mut array: HashMap<(i32, i32), HashMap<i32, f32>> =
            bincode::deserialize(&encoded).unwrap();
        for (_, successors) in &mut array {
            let mut successor_vec: Vec<(i32, f32)> =
                successors.iter().map(|(&k, &v)| (k, v)).collect();
            successor_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top_32: Vec<(i32, f32)> = successor_vec.into_iter().take(12).collect();
            *successors = top_32.into_iter().collect();
        }
        Ok(Reader { array })
    }

    fn search(
        &mut self,
        py_substring: &PyList,
        k: Option<i32>,
        choices: Option<i32>,
        long: Option<i32>,
    ) -> PyResult<(
        Vec<Vec<i32>>,
        Vec<Vec<i32>>,
        Vec<i32>,
        Vec<i32>,
        Vec<Vec<i32>>,
    )> {
        let search_all = Instant::now();
        let mut substring_i32 = Vec::new();
        for item in py_substring.iter() {
            let num: i32 = item.extract()?;
            substring_i32.push(num);
        }
        // add probability to the every 3-gram in the substring
        let mut start = substring_i32.len() as i32 - 16;
        if start < 0 {
            start = 0;
        }
        let start = start as usize; 
        for i in start..substring_i32.len() - 2 {
            let key = (substring_i32[i], substring_i32[i + 1]);
            let count = self.array.entry(key).or_insert(HashMap::new());
            let tt = count.entry(substring_i32[i + 2]).or_insert(0.0);
            *tt += 0.04;
            if *tt > 3.0 {
                *tt = 3.0;
            }
        }
        let mut key = (
            substring_i32[substring_i32.len() - 2],
            substring_i32[substring_i32.len() - 1],
        );
        let mut mcts = MCTS::new(key, &self.array, 4, 24);
        let mut results = mcts.run(150);
        results = results
            .into_iter()
            .filter(|x| x.len() > 1)
            .map(|x| x[1..x.len()].to_vec())
            .collect();

        if results.is_empty() {
            return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()));
        }

        let choices = choices.unwrap_or(24);
        // The items in the heap must be a Trie.

        let verified: Vec<_> = results;
        // Convert into a HashSet to remove duplicates
        let verified: std::collections::HashSet<_> = verified.into_iter().collect();
        let verified: Vec<_> = verified.into_iter().collect();

        let paths = cut_to_choices(verified, choices);
        let (draft_choices, max_branch) = get_draft_choices(paths.clone());
        let (draft_attn_mask, tree_indices, draft_position_ids, retrieve_indices) =
            generate_draft_buffers(draft_choices.clone(), max_branch);

        let max_length = paths.iter().map(|path| path.len()).max().unwrap_or(0);
        let ans = (
            paths
                .into_iter()
                .map(|path| pad_path(path, max_length, -2))
                .collect::<Vec<Vec<i32>>>(),
            draft_attn_mask,
            tree_indices,
            draft_position_ids,
            retrieve_indices,
        );

        Ok(ans)
    }
}

fn cut_to_choices(paths: Vec<Vec<i32>>, choices: i32) -> Vec<Vec<i32>> {
    let mut count: Vec<(usize, usize)> = paths
        .iter()
        .map(|p| {
            (
                p.iter().collect::<std::collections::HashSet<&i32>>().len(),
                paths.iter().position(|x| x == p).unwrap(),
            )
        })
        .collect();
    count.sort_by(|a, b| b.0.cmp(&a.0));

    let mut total_unique = count.iter().map(|(x, _)| x).sum::<usize>();
    let mut to_remove = Vec::new();

    for (c, i) in count {
        if total_unique > choices as usize {
            total_unique -= c;
            to_remove.push(i);
        } else {
            break;
        }
    }

    paths
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !to_remove.contains(i))
        .map(|(_, p)| p)
        .collect()
}

fn get_draft_choices(paths: Vec<Vec<i32>>) -> (Vec<Vec<i32>>, i32) {
    let mut path_dict: HashMap<i32, HashMap<i32, i32>> = HashMap::new();
    let mut cnt_dict: HashMap<i32, i32> = HashMap::new();
    let max_depth = paths.iter().map(|path| path.len() as i32).max().unwrap();

    for depth in 0..max_depth {
        cnt_dict.insert(depth, 0);
    }

    for path in &paths {
        for (depth, item) in path.iter().enumerate() {
            let depth = depth as i32;
            if !path_dict.contains_key(&depth) {
                path_dict.insert(depth, HashMap::new());
            }

            let current_path_dict = path_dict.get_mut(&depth).unwrap();
            if !current_path_dict.contains_key(item) {
                let current_cnt = cnt_dict.get(&depth).unwrap().clone();
                current_path_dict.insert(*item, current_cnt);
                *cnt_dict.get_mut(&depth).unwrap() += 1;
            }
        }
    }

    let max_branch = path_dict.values().map(|v| v.len() as i32).max().unwrap();

    let mut draft_choices: HashSet<Vec<i32>> = HashSet::new();
    for path in paths {
        for (depth, _) in path.iter().enumerate() {
            let depth = depth as i32;
            let draft_choice: Vec<i32> = (0..=depth)
                .map(|prev_depth| {
                    let prev_item = *path.get(prev_depth as usize).unwrap();
                    *path_dict.get(&prev_depth).unwrap().get(&prev_item).unwrap()
                })
                .collect();
            draft_choices.insert(draft_choice);
        }
    }

    let draft_choices: Vec<Vec<i32>> = draft_choices.into_iter().collect();
    (draft_choices, max_branch)
}

fn pad_path(path: Vec<i32>, length: usize, pad_value: i32) -> Vec<i32> {
    let mut path = path;
    while path.len() < length {
        path.push(pad_value);
    }
    path
}

fn generate_draft_buffers(
    draft_choices: Vec<Vec<i32>>,
    topk: i32,
) -> (Vec<Vec<i32>>, Vec<i32>, Vec<i32>, Vec<Vec<i32>>) {
    // Sort the draft_choices based on their lengths and then their values
    let mut sorted_draft_choices = draft_choices;
    sorted_draft_choices.sort_by(|a, b| match a.len().cmp(&b.len()) {
        Ordering::Equal => a.cmp(b),
        other => other,
    });

    sorted_draft_choices =
        sorted_draft_choices[0..std::cmp::min(64, sorted_draft_choices.len())].to_vec();

    let draft_len = sorted_draft_choices.len() + 1;
    assert!(draft_len <= 65, "draft_len should not exceed 65");
    // Initialize depth_counts to keep track of how many choices have a particular depth
    let mut depth_counts: Vec<i32> = vec![0; draft_len];
    let mut prev_depth = 0;
    for path in &sorted_draft_choices {
        let depth = path.len();
        if depth != prev_depth {
            depth_counts[depth - 1] = 0;
        }
        depth_counts[depth - 1] += 1;
        prev_depth = depth;
    }
    // Create the attention mask for draft
    let mut draft_attn_mask: Vec<Vec<i32>> = vec![vec![0; draft_len]; draft_len];
    for i in 0..draft_len {
        draft_attn_mask[i][0] = 1;
        draft_attn_mask[i][i] = 1;
    }

    let mut start = 0;
    for i in 0..depth_counts.len() {
        for j in 0..depth_counts[i] {
            let cur_draft_choice: Vec<i32> = sorted_draft_choices[(start + j) as usize].clone();
            if cur_draft_choice.len() == 1 {
                continue;
            }

            let mut ancestor_idx = vec![];
            for c in 0..(cur_draft_choice.len() - 1) {
                let index = sorted_draft_choices
                    .iter()
                    .position(|x| {
                        x[..=cmp::min(c, x.len() - 1)]
                            == cur_draft_choice[..=cmp::min(c, cur_draft_choice.len() - 1)]
                    })
                    .unwrap()
                    + 1;
                ancestor_idx.push(index);
            }

            for idx in ancestor_idx {
                draft_attn_mask[(j + start + 1) as usize][idx] = 1;
            }
        }
        start += depth_counts[i];
    }

    // Generate tree indices for the draft structure
    let mut draft_tree_indices: Vec<i32> = vec![0; draft_len];
    let mut start = 0;
    for i in 0..depth_counts.len() {
        for j in 0..depth_counts[i] {
            let cur_draft_choice = &sorted_draft_choices[(start + j) as usize];
            draft_tree_indices[(start + j + 1) as usize] =
                cur_draft_choice.last().unwrap() + topk * (i as i32) + 1;
        }
        start += depth_counts[i];
    }

    // Generate position IDs for the draft structure
    let mut draft_position_ids: Vec<i32> = vec![0; draft_len];
    start = 0;
    for i in 0..depth_counts.len() {
        for j in start + 1..start + depth_counts[i] + 1 {
            draft_position_ids[j as usize] = (i as i32) + 1;
        }
        start += depth_counts[i];
    }

    // Generate retrieval indices for draft structure verification
    let mut retrieve_indices_nest = Vec::new();
    let mut retrieve_paths = Vec::new();
    for i in 0..sorted_draft_choices.len() {
        let cur_draft_choice = sorted_draft_choices[sorted_draft_choices.len() - 1 - i].clone();
        let mut retrieve_indice = Vec::new();
        if retrieve_paths.contains(&cur_draft_choice) {
            continue;
        } else {
            for c in 0..cur_draft_choice.len() {
                let index = sorted_draft_choices
                    .iter()
                    .position(|x| *x == cur_draft_choice[0..=c])
                    .unwrap();
                retrieve_indice.push(index as i32);
                retrieve_paths.push(cur_draft_choice[0..=c].to_vec());
            }
        }
        retrieve_indices_nest.push(retrieve_indice);
    }
    let max_length = retrieve_indices_nest.iter().map(|x| x.len()).max().unwrap();
    let mut retrieve_indices: Vec<Vec<i32>> = retrieve_indices_nest
        .iter()
        .map(|x| pad_path(x.clone(), max_length, -2))
        .collect();

    for i in 0..retrieve_indices.len() {
        for j in 0..retrieve_indices[i].len() {
            retrieve_indices[i][j] += 1;
        }
    }

    for i in 0..retrieve_indices.len() {
        retrieve_indices[i].insert(0, 0);
    }

    (
        draft_attn_mask,
        draft_tree_indices,
        draft_position_ids,
        retrieve_indices,
    )
}

#[pymodule]
fn draftretriever(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Writer>()?;
    m.add_class::<Reader>()?;

    Ok(())
}
