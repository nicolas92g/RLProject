use rand::distr::weighted::WeightedIndex;
use rand::prelude::Distribution;

const LINE_WORLD_LENGTH: usize = 5;
const LINE_WORLD_NUM_ACTIONS: usize = 2;
const LINE_WORLD_NUM_TERMINAL_STATES: usize = 2;
const LINE_WORLD_NUM_REWARDS: usize = 3;

const GAMMA_DEFAULT_VALUE:f32 = 0.999999;
const THETA_DEFAULT_VALUE:f32 = 0.000001;

#[allow(nonstandard_style)]
fn iterative_policy_evaluation<
    const NUM_STATES: usize,
    const NUM_ACTIONS: usize,
    const NUM_TERMINAL_STATES: usize,
    const NUM_REWARDS: usize,
>(
    R: [f32; NUM_REWARDS],
    T: [usize; NUM_TERMINAL_STATES],
    p: [[[[f32; NUM_REWARDS]; NUM_STATES]; NUM_ACTIONS]; NUM_STATES],
    pi: [[f32; NUM_ACTIONS]; NUM_STATES],
    gamma: f32,
    theta: f32,
) -> [f32; NUM_STATES] {

    let mut V: [f32; NUM_STATES] = std::array::from_fn(|_| rand::random());
    for s in T {
        V[s] = 0.0
    }

    loop {
        let mut delta = 0.0f32;

        for s_index in 0..NUM_STATES {
            let v = V[s_index];

            let mut total = 0.0f32;
            for a_index in 0..NUM_ACTIONS {
                let mut a_total = 0f32;

                for s_p_index in 0..NUM_STATES {
                    for r_index in 0..NUM_REWARDS {
                        a_total += p[s_index][a_index][s_p_index][r_index] * (R[r_index] + gamma * V[s_p_index]);
                    }
                }

                total += pi[s_index][a_index] * a_total;
            }
            V[s_index] = total;
            delta = delta.max((total - v).abs());
        }

        if delta < theta {
            break;
        }
    }

    V
}

fn my_argmax(collection: impl Iterator<Item=f32>) -> Option<usize> {
    let mut best_index = None;
    let mut best_index_score = 0f32;

    for (index, v) in collection.enumerate() {
        if let Some(_) = best_index && best_index_score >= v {
            continue;
        }
        best_index = Some(index);
        best_index_score = v;
    }

    best_index
}

#[allow(nonstandard_style)]
fn policy_iteration<
    const NUM_STATES: usize,
    const NUM_ACTIONS: usize,
    const NUM_TERMINAL_STATES: usize,
    const NUM_REWARDS: usize,
>(
    R: [f32; NUM_REWARDS],
    T: [usize; NUM_TERMINAL_STATES],
    p: [[[[f32; NUM_REWARDS]; NUM_STATES]; NUM_ACTIONS]; NUM_STATES],
    gamma: f32,
    theta: f32,
) -> ([[f32; NUM_ACTIONS]; NUM_STATES], [f32; NUM_STATES]) {
    let mut V: [f32; NUM_STATES] = std::array::from_fn(|_| rand::random());
    for s in T {
        V[s] = 0.0
    }

    let mut pi: [[f32; NUM_ACTIONS]; NUM_STATES] = std::array::from_fn(|_| {
        let a_rdm_index = rand::random_range(0..NUM_ACTIONS);
        let mut probs = [0f32; NUM_ACTIONS];
        probs[a_rdm_index] = 1.0;
        probs
    });

    loop {
        // Policy Evaluation
        loop {
            let mut delta = 0.0f32;

            for s_index in 0..NUM_STATES {
                let v = V[s_index];

                let mut total = 0.0f32;
                for a_index in 0..NUM_ACTIONS {
                    let mut a_total = 0f32;

                    for s_p_index in 0..NUM_STATES {
                        for r_index in 0..NUM_REWARDS {
                            a_total += p[s_index][a_index][s_p_index][r_index] * (R[r_index] + gamma * V[s_p_index]);
                        }
                    }

                    total += pi[s_index][a_index] * a_total;
                }
                V[s_index] = total;
                delta = delta.max((total - v).abs());
            }

            if delta < theta {
                break;
            }
        }

        // Policy Improvement
        let mut policy_stable = true;
        for s_index in 0..NUM_STATES {
            let old_a = my_argmax(pi[s_index].iter().copied()).unwrap();
            let best_a = my_argmax((0..NUM_ACTIONS).map(|a_index| {
                let mut a_total = 0f32;
                for s_p_index in 0..NUM_STATES {
                    for r_index in 0..NUM_REWARDS {
                        a_total += p[s_index][a_index][s_p_index][r_index] * (R[r_index] + gamma * V[s_p_index]);
                    }
                }
                a_total
            })).unwrap();

            for a in 0..NUM_ACTIONS {
                pi[s_index][a] = if a == best_a { 1.0 } else { 0.0 };
            }
            if old_a != best_a {
                policy_stable = false;
            }
        }

        if policy_stable {
            break;
        }
    }

    (pi, V)
}

trait ModelFreeEnv<const STATES_COUNT: usize, const ACTIONS_COUNT:usize> {
    fn reset(&mut self);
    fn step(&mut self, action: usize);
    fn is_game_over(&self) -> bool;
    fn current_state(&self) -> usize;

    #[allow(unused)]
    fn available_actions(&self) -> impl Iterator<Item=usize>;
    fn score(&self) -> f32;
    fn pretty_print(&self);
}

struct LineWorldEnv<const NUM_CELLS: usize> {
    agent_pos: usize,
}

impl<const NUM_CELLS: usize> LineWorldEnv<NUM_CELLS> {
    pub fn new() -> Self {
        Self {
            agent_pos: NUM_CELLS / 2,
        }
    }
}

impl<const NUM_CELLS: usize> ModelFreeEnv<NUM_CELLS, 2> for LineWorldEnv<NUM_CELLS> {
    fn reset(&mut self) {
        *self = Self::new()
    }

    fn step(&mut self, action: usize) {
        assert!(!self.is_game_over());

        match action {
            0 => self.agent_pos -= 1,
            1 => self.agent_pos += 1,
            _ => panic!("Invalid action !")
        }
    }

    fn is_game_over(&self) -> bool {
        self.agent_pos == 0 || self.agent_pos == NUM_CELLS - 1
    }

    fn current_state(&self) -> usize {
        self.agent_pos
    }

    fn available_actions(&self) -> impl Iterator<Item=usize> {
        [0, 1].into_iter()
    }

    fn score(&self) -> f32 {
        match self.agent_pos {
            0 => -1.0f32,
            pos if pos == NUM_CELLS - 1 => 1.0f32,
            _ => 0f32
        }
    }

    fn pretty_print(&self) {
        for s in 0..NUM_CELLS {
            if s == self.agent_pos {
                print!("X");
            } else {
                print!("_");
            }
        }
        println!();
    }
}

#[allow(nonstandard_style)]
fn first_visit_monte_carlo_prediction<
    T: ModelFreeEnv<STATES_COUNT, ACTIONS_COUNT>,
    const STATES_COUNT: usize,
    const ACTIONS_COUNT: usize>(
    mut env: T,
    pi: [[f32; ACTIONS_COUNT]; STATES_COUNT],
    iterations_count: usize,
    gamma: f32,
) -> [f32; STATES_COUNT] {
    let mut V = std::array::from_fn(|_| rand::random());
    let mut V_counts:[usize; STATES_COUNT] = std::array::from_fn(|_| 0usize);

    let mut trajectory_states = vec![];
    let mut trajectory_actions = vec![];
    let mut trajectory_rewards = vec![];


    for _it in 0..iterations_count {
        env.reset();

        trajectory_states.clear();
        trajectory_actions.clear();
        trajectory_rewards.clear();

        while !env.is_game_over() {
            let s = env.current_state();
            let a_probs = pi[s];
            let a = WeightedIndex::new(&a_probs).unwrap().sample(&mut rand::rng());

            let prev_score = env.score();
            env.step(a);
            let r = env.score() - prev_score;

            trajectory_states.push(s);
            trajectory_actions.push(a);
            trajectory_rewards.push(r);
        }

        V[env.current_state()] = 0f32;

        let mut G = 0f32;
        for (t, ((&s, _), &r)) in trajectory_states.iter().zip(trajectory_actions.iter()).zip(trajectory_rewards.iter()).enumerate().rev() {
            G = gamma * G + r;
            if !trajectory_states[..t].contains(&s) {
                let count_f32 = V_counts[s] as f32;
                V[s] = (V[s] * f32::from(count_f32) + G) / (count_f32 + 1f32);
                V_counts[s] += 1;
            }
        }
    }
    V
}

#[allow(nonstandard_style)]
fn on_policy_monte_carlo_control<
    T: ModelFreeEnv<STATES_COUNT, ACTIONS_COUNT>,
    const STATES_COUNT: usize,
    const ACTIONS_COUNT: usize>(
    mut env: T,
    iterations_count: usize,
    gamma: f32,
    epsilon: f32,
) -> ([[f32; ACTIONS_COUNT]; STATES_COUNT], [[f32; ACTIONS_COUNT]; STATES_COUNT]) {
    let mut Q = std::array::from_fn(|_| std::array::from_fn(|_| rand::random()));
    let mut Q_counts:[[usize; ACTIONS_COUNT]; STATES_COUNT] = std::array::from_fn(|_| std::array::from_fn(|_| 0));

    let mut pi = std::array::from_fn(|_| std::array::from_fn(|_| 1f32 / ACTIONS_COUNT as f32));

    let mut trajectory_states = vec![];
    let mut trajectory_actions = vec![];
    let mut trajectory_rewards = vec![];


    for _it in 0..iterations_count {
        env.reset();

        trajectory_states.clear();
        trajectory_actions.clear();
        trajectory_rewards.clear();

        while !env.is_game_over() {
            let s = env.current_state();
            let a_probs = pi[s];
            let a = WeightedIndex::new(&a_probs).unwrap().sample(&mut rand::rng());

            let prev_score = env.score();
            env.step(a);
            let r = env.score() - prev_score;

            trajectory_states.push(s);
            trajectory_actions.push(a);
            trajectory_rewards.push(r);
        }

        for a in 0..ACTIONS_COUNT {
            Q[env.current_state()][a] = 0f32;
        }

        let mut G = 0f32;
        for (t, ((&s, &a), &r)) in trajectory_states.iter().zip(trajectory_actions.iter()).zip(trajectory_rewards.iter()).enumerate().rev() {
            G = gamma * G + r;
            if !trajectory_states[..t].contains(&s) {
                let count_f32 = Q_counts[s][a] as f32;
                Q[s][a] = (Q[s][a] * f32::from(count_f32) + G) / (count_f32 + 1f32);
                Q_counts[s][a] += 1;

                let best_a = my_argmax(Q[s].iter().copied()).unwrap();

                for a in 0..ACTIONS_COUNT {
                    pi[s][a] = epsilon / ACTIONS_COUNT as f32;
                }
                pi[s][best_a] = 1f32 - epsilon + epsilon / ACTIONS_COUNT as f32;
            }
        }
    }
    (pi, Q)
}

#[allow(nonstandard_style)]
fn main() {
    let R: [f32; LINE_WORLD_NUM_REWARDS] = [0.0, -1.0, 1.0]; // Rewards
    let T: [usize; LINE_WORLD_NUM_TERMINAL_STATES] = [0, LINE_WORLD_LENGTH - 1]; // Terminal States

    let mut p = [[[[0.0f32; LINE_WORLD_NUM_REWARDS]; LINE_WORLD_LENGTH]; LINE_WORLD_NUM_ACTIONS];
        LINE_WORLD_LENGTH];

    p[1][0][0][1] = 1.0;
    p[LINE_WORLD_LENGTH - 2][1][LINE_WORLD_LENGTH - 1][2] = 1.0;

    for s in 2..LINE_WORLD_LENGTH - 1 {
        p[s][0][s - 1][0] = 1.0
    }

    for s in 1..LINE_WORLD_LENGTH - 2 {
        p[s][1][s + 1][0] = 1.0
    }

    let pi_right = std::array::from_fn(|_| [0.0, 1.0]);

    let V = iterative_policy_evaluation(R, T, p, pi_right,
                                        GAMMA_DEFAULT_VALUE,
                                        THETA_DEFAULT_VALUE);

    dbg!(V);

    let pi_left = std::array::from_fn(|_| [1.0, 0.0]);

    let V = iterative_policy_evaluation(R, T, p, pi_left,
                                        GAMMA_DEFAULT_VALUE,
                                        THETA_DEFAULT_VALUE);

    dbg!(V);

    let pi_uniformly_random = std::array::from_fn(|_| [0.5, 0.5]);

    let V = iterative_policy_evaluation(R, T, p, pi_uniformly_random,
                                        GAMMA_DEFAULT_VALUE,
                                        THETA_DEFAULT_VALUE);

    dbg!(V);

    let pi_mostly_right = std::array::from_fn(|_| [0.2, 0.8]);

    let V = iterative_policy_evaluation(R, T, p, pi_mostly_right,
                                        GAMMA_DEFAULT_VALUE,
                                        THETA_DEFAULT_VALUE);

    dbg!(V);

    let now = std::time::Instant::now();
    let (pi, v) = policy_iteration(R, T, p, GAMMA_DEFAULT_VALUE, THETA_DEFAULT_VALUE);
    println!("Elapsed : {}", now.elapsed().as_secs_f64());

    dbg!((pi, v));

    let mut env = LineWorldEnv::<5>::new();
    env.pretty_print();

    env.step(1);
    env.pretty_print();

    env.step(1);
    env.pretty_print();

    dbg!(env.score());

    env.reset();
    env.pretty_print();

    let now = std::time::Instant::now();
    let V = first_visit_monte_carlo_prediction(LineWorldEnv::<LINE_WORLD_LENGTH>::new(), pi_right, 10_000, GAMMA_DEFAULT_VALUE);
    println!("Elapsed : {}", now.elapsed().as_secs_f64());
    dbg!(V);

    let now = std::time::Instant::now();
    let (pi, Q) = on_policy_monte_carlo_control(LineWorldEnv::<LINE_WORLD_LENGTH>::new(), 10_000, GAMMA_DEFAULT_VALUE, 0.05);
    println!("Elapsed : {}", now.elapsed().as_secs_f64());
    dbg!(pi);
    dbg!(Q);
}

