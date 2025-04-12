use std::collections::{vec_deque::{Iter, IterMut}, VecDeque};

pub struct MessageQueue {
    messages: VecDeque<String>,
    max_size: usize,
}

impl MessageQueue {
    pub fn new(max_size: usize) -> Self {
        assert!(max_size > 0);
        Self {
            max_size,
            messages: VecDeque::with_capacity(max_size),
        }
    }

    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn push<T: std::fmt::Display>(&mut self, message: T) {
        if self.messages.len() == self.max_size {
            self.messages.pop_front();
        }
        self.messages.push_back(message.to_string());
    }

    pub fn iter<'a>(&'a self) -> Iter<'a, String> {
        self.messages.iter()
    }

    pub fn iter_mut<'a>(&'a mut self) -> IterMut<'a, String> {
        self.messages.iter_mut()
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }
}

impl std::ops::Index<usize> for MessageQueue {
    type Output = String;
    fn index(&self, index: usize) -> &Self::Output {
        &self.messages[index]
    }
}

impl std::ops::IndexMut<usize> for MessageQueue {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.messages[index]
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn message_queue_test() {
        let mut messages = MessageQueue::new(5);
        for i in 0..10 {
            messages.push(format!("Hello, world! {i}"));
        }
        messages.iter().for_each(|msg| {
            println!("{}", msg);
        });
        messages.push(1234);
        println!("***");
        messages.iter().for_each(|msg| {
            println!("{}", msg);
        });
    }
}