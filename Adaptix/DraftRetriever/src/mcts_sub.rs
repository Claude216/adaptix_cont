use rand::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::Hash;
use std::rc::Rc;

const c1: f32 = 32.0;
const c2: f32 = 8.0;

#[derive(Debug, Clone)]
pub struct Node {
    word: (i32, i32),
    parent: Option<Rc<RefCell<Node>>>,
    children: Vec<Rc<RefCell<Node>>>,
    visits: u32,
    score: f32,
    untried_words: Vec<i32>,
    p: f32,
}

impl Node {
    pub fn new(
        word: (i32, i32),
        parent: Option<Rc<RefCell<Node>>>,
        untried_words: Vec<i32>,
        p: f32,
    ) -> Self {
        Node {
            word,
            parent,
            children: Vec::new(),
            visits: 0,
            score: 0.0,
            untried_words,
            p,
        }
    }

    pub fn putc_score(&self, total_visits: u32) -> f32 {
        if self.visits == 0 {
            return std::f32::INFINITY;
        }
        self.score / self.visits as f32
            + (c1 + ((total_visits as f32 + c2 + 1.0) / c2).ln())
                * self.p
                * (total_visits as f32).sqrt()
                / (1 + self.visits) as f32
    }

    pub fn select_child(&self) -> Rc<RefCell<Node>> {
        let mut max_score = 0f32;
        let mut max_child = self.children[0].clone();
        for child in self.children.iter() {
            let child_score = child.borrow().putc_score(self.visits);
            if child_score > max_score {
                max_score = child_score;
                max_child = child.clone();
            }
        }
        max_child
    }
}

pub struct MCTS<'a> {
    root: Rc<RefCell<Node>>,
    trigram_matrix: &'a HashMap<(i32, i32), HashMap<i32, f32>>,
    sentence_length: usize,
    top_k: usize,
}

impl<'a> MCTS<'a> {
    pub fn new(
        start_word: (i32, i32),
        trigram_matrix: &'a HashMap<(i32, i32), HashMap<i32, f32>>,
        sentence_length: usize,
        top_k: usize,
    ) -> Self {
        let untried_words: Vec<i32> = trigram_matrix
            .get(&start_word)
            .unwrap_or(&HashMap::new())
            .keys()
            .cloned()
            .collect();

        let root_node = Node::new(start_word, None, untried_words, 0.0);
        MCTS {
            root: Rc::new(RefCell::new(root_node)),
            trigram_matrix,
            sentence_length,
            top_k,
        }
    }

    pub fn run(&mut self, iterations: i32) -> Vec<Vec<i32>> {
        let mut rng = rand::thread_rng();
        for _ in 0..iterations {
            let mut node = self.root.clone();
            let mut state: Vec<(i32, i32)> = vec![node.borrow().word];
            let mut end = false;
            while !node.borrow().untried_words.is_empty()
                && node.borrow().children.len() > 0
                && !end
            {
                let selected_child = {
                    let n = node.borrow();
                    n.select_child()
                };
                state.push(selected_child.borrow().word);
                node = selected_child;
                if state.len() == self.sentence_length {
                    end = true;
                }
            }
            if !node.borrow().untried_words.is_empty() && !end {
                let untried_words = node.borrow().untried_words.clone();
                let p_word = node.borrow().word.1;
                let k = node.borrow().word.clone();

                for word in untried_words {
                    let tmp_untried_words: Vec<i32> = self
                        .trigram_matrix
                        .get(&(p_word, word))
                        .unwrap_or(&HashMap::new())
                        .keys()
                        .cloned()
                        .collect();
                    let child_node = Node::new(
                        (p_word, word),
                        Some(node.clone()),
                        tmp_untried_words,
                        *self
                            .trigram_matrix
                            .get(&k)
                            .unwrap_or(&HashMap::new())
                            .get(&word)
                            .unwrap_or(&0f32),
                    );
                    node.borrow_mut()
                        .children
                        .push(Rc::new(RefCell::new(child_node)));
                }
                let new_child = node.borrow().children.last().unwrap().clone();
                node = new_child;
                state.push(node.borrow().word);
            }
            while state.len() < self.sentence_length {
                let last_word = *state.last().unwrap();
                if let Some(next_words) = self.trigram_matrix.get(&last_word) {
                    // Collect next words and their scores
                    let mut word_scores: Vec<(i32, f32)> = next_words
                        .iter()
                        .map(|(&word, &score)| (word, score))
                        .collect();

                    if word_scores.is_empty() {
                        break;
                    }

                    // Normalize scores to get probabilities
                    let total_score: f32 = word_scores.iter().map(|(_, score)| score).sum();
                    let probabilities: Vec<f32> = word_scores
                        .iter()
                        .map(|(_, score)| score / total_score)
                        .collect();

                    // Choose the next word based on the weighted probabilities
                    let mut cumulative_probabilities = vec![0f32; probabilities.len()];
                    let mut sum = 0f32;
                    for (i, &prob) in probabilities.iter().enumerate() {
                        sum += prob;
                        cumulative_probabilities[i] = sum;
                    }

                    // Generate a random number between 0 and 1

                    let choice = rng.gen::<f32>();

                    // Find the word corresponding to the random choice
                    let mut chosen_word = word_scores.last().unwrap().0;

                    for (i, &cumulative_prob) in cumulative_probabilities.iter().enumerate() {
                        if choice <= cumulative_prob {
                            chosen_word = word_scores[i].0;
                            break;
                        }
                    }

                    // Push the chosen word onto the state
                    state.push((last_word.1, chosen_word));
                } else {
                    break;
                }
            }

            // Backpropagation
            let mut score: f32 = 1.0;
            for i in 0..state.len() - 1 {
                score += self
                    .trigram_matrix
                    .get(&state[i])
                    .unwrap()
                    .get(&state[i + 1].0)
                    .cloned()
                    .unwrap_or(0.0);
            }

            let mut current_node = Some(node);
            while let Some(n) = current_node {
                let mut n_ref = n.borrow_mut();
                n_ref.visits += 1;
                n_ref.score += score;
                if let Some(parent) = n_ref.parent.clone() {
                    current_node = Some(parent);
                } else {
                    break;
                }
            }
        }

        // Extract the top k sentences
        let mut sentences = Vec::new();

        self.extract_sentences(&mut sentences, &self.root, Vec::new());
        sentences.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut sentences: Vec<Vec<i32>> = sentences
            .iter()
            .take(self.top_k)
            .map(|(s, _)| s.clone())
            .collect();
        for sentence in sentences.iter_mut() {
            if sentence.len() >= 2 {
                let mut second_last = sentence[sentence.len() - 2];
                let mut last = sentence[sentence.len() - 1];
                if let Some(next_words_probs) = self.trigram_matrix.get(&(second_last, last)) {
                    if let Some((&next_word, &next_word_prob)) = next_words_probs
                        .iter()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    {
                        sentence.push(next_word);
                        second_last = last;
                        last = next_word;
                    }
                }
            }
        }
        sentences
    }

    pub fn extract_sentences(
        &self,
        sentences: &mut Vec<(Vec<i32>, f32)>,
        node: &Rc<RefCell<Node>>,
        current_sentence: Vec<i32>,
    ) {
        let n = node.borrow();
        let mut sentence = current_sentence.clone();
        sentence.push(n.word.1);
        sentences.push((sentence.clone(), n.visits as f32));
        if sentence.len() > self.sentence_length || n.children.is_empty() {
            return;
        }

        for child in n.children.iter() {
            self.extract_sentences(sentences, child, sentence.clone());
        }
    }
}
