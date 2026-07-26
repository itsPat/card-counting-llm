#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace {

constexpr int kAce = 0;
constexpr int kTen = 9;
constexpr int kMaximumHands = 4;
constexpr int kMinimumProfitHalfUnits = -17;
constexpr int kMaximumProfitHalfUnits = 18;
constexpr int kOutcomeCount =
    kMaximumProfitHalfUnits - kMinimumProfitHalfUnits + 1;

using Counts = std::array<int, 10>;
using Histogram = std::array<std::uint64_t, kOutcomeCount>;

constexpr std::uint8_t kFromSplit = 1U << 0U;
constexpr std::uint8_t kSplitAces = 1U << 1U;
constexpr std::uint8_t kCanDouble = 1U << 2U;
constexpr std::uint8_t kCanSurrender = 1U << 3U;
constexpr std::uint8_t kSurrendered = 1U << 4U;
constexpr std::uint8_t kFinished = 1U << 5U;

class SplitMix64 {
public:
    explicit SplitMix64(const std::uint64_t seed) : state_(seed) {}

    std::uint64_t next() {
        std::uint64_t value = (state_ += 0x9e3779b97f4a7c15ULL);
        value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31U);
    }

    std::uint64_t uniform_below(const std::uint64_t bound) {
        const std::uint64_t threshold =
            static_cast<std::uint64_t>(-bound) % bound;
        while (true) {
            const std::uint64_t value = next();
            if (value >= threshold) {
                return value % bound;
            }
        }
    }

private:
    std::uint64_t state_;
};

struct Shoe {
    Counts counts{};
    int total = 0;

    int draw(SplitMix64& random) {
        const auto selected = static_cast<int>(
            random.uniform_below(static_cast<std::uint64_t>(total))
        );
        int cumulative = 0;
        for (int card = 0; card < 10; ++card) {
            cumulative += counts[card];
            if (selected < cumulative) {
                --counts[card];
                --total;
                return card;
            }
        }
        return -1;
    }
};

struct Hand {
    int hard_total = 0;
    int aces = 0;
    int card_count = 0;
    int first_card = -1;
    int second_card = -1;
    int wager_half_units = 2;
    bool from_split = false;
    bool split_aces = false;
    bool can_double = true;
    bool can_surrender = true;
    bool surrendered = false;
    bool finished = false;
};

enum class Action {
    hit,
    stand,
    double_down,
    split,
    surrender,
};

int hard_value(const int card) {
    return card == kAce ? 1 : card + 1;
}

void add_card(Hand& hand, const int card) {
    hand.hard_total += hard_value(card);
    hand.aces += card == kAce ? 1 : 0;
    if (hand.card_count == 0) {
        hand.first_card = card;
    } else if (hand.card_count == 1) {
        hand.second_card = card;
    }
    ++hand.card_count;
}

Hand two_card_hand(
    const int first,
    const int second,
    const int wager_half_units = 2,
    const bool from_split = false,
    const bool split_aces = false
) {
    Hand hand{};
    hand.wager_half_units = wager_half_units;
    hand.from_split = from_split;
    hand.split_aces = split_aces;
    hand.can_surrender = !from_split;
    add_card(hand, first);
    add_card(hand, second);
    return hand;
}

int hand_total(const Hand& hand) {
    return hand.aces > 0 && hand.hard_total + 10 <= 21
        ? hand.hard_total + 10
        : hand.hard_total;
}

bool is_soft(const Hand& hand) {
    return hand.aces > 0 && hand.hard_total + 10 <= 21;
}

bool is_bust(const Hand& hand) {
    return hand_total(hand) > 21;
}

bool is_natural_blackjack(const Hand& hand) {
    return !hand.from_split && hand.card_count == 2 &&
        hand_total(hand) == 21;
}

bool is_pair(const Hand& hand) {
    return hand.card_count == 2 && hand.first_card == hand.second_card;
}

int dealer_value(const int card) {
    return card == kAce ? 11 : hard_value(card);
}

bool contains(const int value, const std::initializer_list<int> choices) {
    for (const int choice : choices) {
        if (value == choice) {
            return true;
        }
    }
    return false;
}

Action basic_strategy_action(
    const Hand& hand,
    const int dealer_upcard,
    const int hands_in_round
) {
    const int total = hand_total(hand);
    const int dealer = dealer_value(dealer_upcard);
    const bool can_double = hand.card_count == 2 && hand.can_double;
    const bool can_split =
        is_pair(hand) && hands_in_round < kMaximumHands &&
        !(hand.from_split && hand.first_card == kAce);
    const bool can_surrender =
        !hand.from_split && hand.card_count == 2 && hand.can_surrender;

    if (can_surrender && !is_soft(hand)) {
        const bool surrender =
            (total == 17 && dealer == 11) ||
            (total == 16 && contains(dealer, {9, 10, 11})) ||
            (total == 15 && contains(dealer, {10, 11}));
        if (surrender) {
            return Action::surrender;
        }
    }

    if (can_split) {
        const int pair = hand.first_card;
        const bool split =
            pair == kAce || pair == 7 ||
            (pair == 8 && contains(dealer, {2, 3, 4, 5, 6, 8, 9})) ||
            (pair == 6 && contains(dealer, {2, 3, 4, 5, 6, 7})) ||
            (pair == 5 && contains(dealer, {2, 3, 4, 5, 6})) ||
            (pair == 3 && contains(dealer, {5, 6})) ||
            ((pair == 1 || pair == 2) &&
             contains(dealer, {2, 3, 4, 5, 6, 7}));
        if (split) {
            return Action::split;
        }
    }

    if (is_soft(hand)) {
        if (total >= 20) {
            return Action::stand;
        }
        if (total == 19) {
            return can_double && dealer == 6
                ? Action::double_down
                : Action::stand;
        }
        if (total == 18) {
            if (can_double && contains(dealer, {2, 3, 4, 5, 6})) {
                return Action::double_down;
            }
            return contains(dealer, {2, 3, 4, 5, 6, 7, 8})
                ? Action::stand
                : Action::hit;
        }
        const bool soft_double =
            (total == 17 && contains(dealer, {3, 4, 5, 6})) ||
            ((total == 15 || total == 16) &&
             contains(dealer, {4, 5, 6})) ||
            ((total == 13 || total == 14) &&
             contains(dealer, {5, 6}));
        return can_double && soft_double ? Action::double_down : Action::hit;
    }

    if (total >= 17) {
        return Action::stand;
    }
    if (total >= 13) {
        return contains(dealer, {2, 3, 4, 5, 6})
            ? Action::stand
            : Action::hit;
    }
    if (total == 12) {
        return contains(dealer, {4, 5, 6}) ? Action::stand : Action::hit;
    }
    if (can_double && total == 11) {
        return Action::double_down;
    }
    if (can_double && total == 10 && dealer >= 2 && dealer <= 9) {
        return Action::double_down;
    }
    if (can_double && total == 9 && contains(dealer, {3, 4, 5, 6})) {
        return Action::double_down;
    }
    return Action::hit;
}

bool dealer_should_hit(const Hand& dealer) {
    const int total = hand_total(dealer);
    return total < 17 || (total == 17 && is_soft(dealer));
}

int profit_against_dealer(const Hand& player, const Hand& dealer) {
    if (player.surrendered) {
        return -player.wager_half_units / 2;
    }
    if (is_bust(player)) {
        return -player.wager_half_units;
    }
    if (is_bust(dealer) || hand_total(player) > hand_total(dealer)) {
        return player.wager_half_units;
    }
    if (hand_total(player) == hand_total(dealer)) {
        return 0;
    }
    return -player.wager_half_units;
}

bool hole_is_allowed(
    const int hole,
    const int dealer_upcard,
    const bool negative_peek
) {
    if (!negative_peek) {
        return true;
    }
    if (dealer_upcard == kAce) {
        return hole != kTen;
    }
    if (dealer_upcard == kTen) {
        return hole != kAce;
    }
    return true;
}

int draw_hole(
    Shoe& shoe,
    const int dealer_upcard,
    const bool negative_peek,
    SplitMix64& random
) {
    int eligible = 0;
    for (int card = 0; card < 10; ++card) {
        if (hole_is_allowed(card, dealer_upcard, negative_peek)) {
            eligible += shoe.counts[card];
        }
    }
    if (eligible == 0) {
        return -1;
    }
    const auto selected = static_cast<int>(
        random.uniform_below(static_cast<std::uint64_t>(eligible))
    );
    int cumulative = 0;
    for (int card = 0; card < 10; ++card) {
        if (!hole_is_allowed(card, dealer_upcard, negative_peek)) {
            continue;
        }
        cumulative += shoe.counts[card];
        if (selected < cumulative) {
            --shoe.counts[card];
            --shoe.total;
            return card;
        }
    }
    return -1;
}

void split_active_hand(
    std::array<Hand, kMaximumHands>& hands,
    int& hand_count,
    const int active,
    Shoe& shoe,
    SplitMix64& random
) {
    const Hand original = hands[active];
    const int pair = original.first_card;
    const bool split_aces = pair == kAce;
    const Hand left = two_card_hand(
        pair,
        shoe.draw(random),
        original.wager_half_units,
        true,
        split_aces
    );
    const Hand right = two_card_hand(
        pair,
        shoe.draw(random),
        original.wager_half_units,
        true,
        split_aces
    );
    for (int index = hand_count; index > active + 1; --index) {
        hands[index] = hands[index - 1];
    }
    hands[active] = left;
    hands[active + 1] = right;
    ++hand_count;
}

void apply_action(
    std::array<Hand, kMaximumHands>& hands,
    int& hand_count,
    const int active,
    const Action action,
    Shoe& shoe,
    SplitMix64& random
) {
    Hand& hand = hands[active];
    if (action == Action::hit) {
        add_card(hand, shoe.draw(random));
        hand.can_double = false;
        hand.can_surrender = false;
    } else if (action == Action::stand) {
        hand.finished = true;
    } else if (action == Action::double_down) {
        hand.wager_half_units *= 2;
        add_card(hand, shoe.draw(random));
        hand.can_double = false;
        hand.can_surrender = false;
        hand.finished = true;
    } else if (action == Action::surrender) {
        hand.surrendered = true;
        hand.finished = true;
    } else {
        split_active_hand(hands, hand_count, active, shoe, random);
    }
}

Action action_from_code(const int action) {
    return static_cast<Action>(action);
}

int settle_hands(
    std::array<Hand, kMaximumHands>& hands,
    const int hand_count,
    Hand& dealer,
    Shoe& shoe,
    SplitMix64& random
) {
    bool all_terminal_losses = true;
    int terminal_profit = 0;
    for (int index = 0; index < hand_count; ++index) {
        const Hand& hand = hands[index];
        if (!hand.surrendered && !is_bust(hand)) {
            all_terminal_losses = false;
        }
        if (hand.surrendered) {
            terminal_profit -= hand.wager_half_units / 2;
        } else if (is_bust(hand)) {
            terminal_profit -= hand.wager_half_units;
        }
    }
    if (all_terminal_losses) {
        return terminal_profit;
    }

    while (dealer_should_hit(dealer)) {
        add_card(dealer, shoe.draw(random));
    }
    int profit = 0;
    for (int index = 0; index < hand_count; ++index) {
        profit += profit_against_dealer(hands[index], dealer);
    }
    return profit;
}

int simulate_play_action(
    const Counts& initial_counts,
    const int initial_total,
    const int unseen_unavailable,
    const int dealer_upcard,
    const bool negative_peek,
    const std::array<Hand, kMaximumHands>& initial_hands,
    const int initial_hand_count,
    const int initial_active,
    const Action forced_action,
    SplitMix64& random
) {
    Shoe shoe{initial_counts, initial_total};
    const int dealer_hole =
        draw_hole(shoe, dealer_upcard, negative_peek, random);
    if (dealer_hole < 0) {
        return std::numeric_limits<int>::min();
    }
    const int other_unavailable =
        unseen_unavailable > 0 ? unseen_unavailable - 1 : 0;
    for (int index = 0; index < other_unavailable; ++index) {
        if (shoe.total == 0) {
            return std::numeric_limits<int>::min();
        }
        static_cast<void>(shoe.draw(random));
    }

    Hand dealer = two_card_hand(dealer_upcard, dealer_hole);
    std::array<Hand, kMaximumHands> hands = initial_hands;
    int hand_count = initial_hand_count;
    apply_action(
        hands,
        hand_count,
        initial_active,
        forced_action,
        shoe,
        random
    );

    int active = initial_active;
    while (active < hand_count) {
        Hand& hand = hands[active];
        if (hand.finished || is_bust(hand) || hand_total(hand) >= 21 ||
            hand.split_aces) {
            hand.finished = true;
            ++active;
            continue;
        }
        const Action action =
            basic_strategy_action(hand, dealer_upcard, hand_count);
        apply_action(hands, hand_count, active, action, shoe, random);
    }
    return settle_hands(hands, hand_count, dealer, shoe, random);
}

int simulate_round(
    const Counts& initial_counts,
    const int initial_total,
    const int unseen_unavailable,
    SplitMix64& random
) {
    Shoe shoe{initial_counts, initial_total};
    for (int index = 0; index < unseen_unavailable; ++index) {
        static_cast<void>(shoe.draw(random));
    }

    Counts public_counts = initial_counts;
    int public_total = initial_total;
    const int first_player = shoe.draw(random);
    --public_counts[first_player];
    --public_total;
    const int dealer_upcard = shoe.draw(random);
    --public_counts[dealer_upcard];
    --public_total;
    const int second_player = shoe.draw(random);
    --public_counts[second_player];
    --public_total;

    const bool take_insurance =
        dealer_upcard == kAce && 3 * public_counts[kTen] > public_total;
    const int dealer_hole = shoe.draw(random);
    Hand dealer = two_card_hand(dealer_upcard, dealer_hole);
    Hand player = two_card_hand(first_player, second_player);
    const bool dealer_blackjack = hand_total(dealer) == 21;
    const int insurance_profit = !take_insurance
        ? 0
        : (dealer_blackjack ? 2 : -1);

    if (dealer_blackjack) {
        const int player_profit = is_natural_blackjack(player) ? 0 : -2;
        return player_profit + insurance_profit;
    }
    if (is_natural_blackjack(player)) {
        return 3 + insurance_profit;
    }

    std::array<Hand, kMaximumHands> hands{};
    hands[0] = player;
    int hand_count = 1;
    int active = 0;
    while (active < hand_count) {
        Hand& hand = hands[active];
        if (hand.finished || is_bust(hand) || hand_total(hand) >= 21 ||
            hand.split_aces) {
            hand.finished = true;
            ++active;
            continue;
        }

        const Action action =
            basic_strategy_action(hand, dealer_upcard, hand_count);
        if (action == Action::hit) {
            add_card(hand, shoe.draw(random));
        } else if (action == Action::stand) {
            hand.finished = true;
        } else if (action == Action::double_down) {
            hand.wager_half_units *= 2;
            add_card(hand, shoe.draw(random));
            hand.finished = true;
        } else if (action == Action::surrender) {
            hand.surrendered = true;
            hand.finished = true;
        } else {
            const int pair = hand.first_card;
            const int wager = hand.wager_half_units;
            const bool split_aces = pair == kAce;
            const Hand left = two_card_hand(
                pair,
                shoe.draw(random),
                wager,
                true,
                split_aces
            );
            const Hand right = two_card_hand(
                pair,
                shoe.draw(random),
                wager,
                true,
                split_aces
            );
            for (int index = hand_count; index > active + 1; --index) {
                hands[index] = hands[index - 1];
            }
            hands[active] = left;
            hands[active + 1] = right;
            ++hand_count;
        }
    }

    bool all_terminal_losses = true;
    int terminal_profit = insurance_profit;
    for (int index = 0; index < hand_count; ++index) {
        const Hand& hand = hands[index];
        if (!hand.surrendered && !is_bust(hand)) {
            all_terminal_losses = false;
        }
        if (hand.surrendered) {
            terminal_profit -= hand.wager_half_units / 2;
        } else if (is_bust(hand)) {
            terminal_profit -= hand.wager_half_units;
        }
    }
    if (all_terminal_losses) {
        return terminal_profit;
    }

    while (dealer_should_hit(dealer)) {
        add_card(dealer, shoe.draw(random));
    }
    int profit = insurance_profit;
    for (int index = 0; index < hand_count; ++index) {
        profit += profit_against_dealer(hands[index], dealer);
    }
    return profit;
}

}  // namespace

extern "C" int blackjack_fixed_policy_simulation(
    const int* raw_counts,
    const int unseen_unavailable,
    const std::uint64_t seed,
    const std::uint64_t rollouts,
    std::uint64_t* output
) {
    if (raw_counts == nullptr || output == nullptr ||
        unseen_unavailable < 0 || rollouts == 0) {
        return 1;
    }
    Counts counts{};
    int total = 0;
    for (int index = 0; index < 10; ++index) {
        if (raw_counts[index] < 0) {
            return 2;
        }
        counts[index] = raw_counts[index];
        total += raw_counts[index];
    }
    if (total - unseen_unavailable < 22) {
        return 3;
    }
    for (int index = 0; index < kOutcomeCount; ++index) {
        output[index] = 0;
    }

    SplitMix64 random(seed);
    for (std::uint64_t rollout = 0; rollout < rollouts; ++rollout) {
        const int profit =
            simulate_round(counts, total, unseen_unavailable, random);
        if (profit < kMinimumProfitHalfUnits ||
            profit > kMaximumProfitHalfUnits) {
            return 4;
        }
        ++output[profit - kMinimumProfitHalfUnits];
    }
    return 0;
}

extern "C" int blackjack_play_action_simulation(
    const int* raw_counts,
    const int unseen_unavailable,
    const int dealer_upcard,
    const int negative_peek,
    const int* raw_hand_counts,
    const int* raw_hand_totals,
    const int* raw_wagers,
    const std::uint8_t* raw_hand_flags,
    const int hand_count,
    const int active_hand,
    const int forced_action,
    const std::uint64_t seed,
    const std::uint64_t rollouts,
    std::uint64_t* output
) {
    if (raw_counts == nullptr || raw_hand_counts == nullptr ||
        raw_hand_totals == nullptr || raw_wagers == nullptr ||
        raw_hand_flags == nullptr || output == nullptr ||
        unseen_unavailable < 0 || dealer_upcard < 0 || dealer_upcard > 9 ||
        (negative_peek != 0 && negative_peek != 1) ||
        hand_count < 1 || hand_count > kMaximumHands ||
        active_hand < 0 || active_hand >= hand_count ||
        forced_action < 0 || forced_action > 4 || rollouts == 0) {
        return 1;
    }

    Counts counts{};
    int total = 0;
    for (int index = 0; index < 10; ++index) {
        if (raw_counts[index] < 0) {
            return 2;
        }
        counts[index] = raw_counts[index];
        total += raw_counts[index];
    }
    if (total <= unseen_unavailable || total < 16) {
        return 3;
    }

    std::array<Hand, kMaximumHands> hands{};
    for (int hand_index = 0; hand_index < hand_count; ++hand_index) {
        if (raw_wagers[hand_index] <= 0) {
            return 4;
        }
        Hand hand{};
        hand.wager_half_units = raw_wagers[hand_index];
        const std::uint8_t flags = raw_hand_flags[hand_index];
        hand.from_split = (flags & kFromSplit) != 0;
        hand.split_aces = (flags & kSplitAces) != 0;
        hand.can_double = (flags & kCanDouble) != 0;
        hand.can_surrender = (flags & kCanSurrender) != 0;
        hand.surrendered = (flags & kSurrendered) != 0;
        hand.finished = (flags & kFinished) != 0;
        if (hand.finished) {
            if (raw_hand_totals[hand_index] < 0) {
                return 5;
            }
            hand.hard_total = raw_hand_totals[hand_index];
        } else {
            for (int card = 0; card < 10; ++card) {
                const int card_count =
                    raw_hand_counts[hand_index * 10 + card];
                if (card_count < 0) {
                    return 6;
                }
                for (int occurrence = 0; occurrence < card_count; ++occurrence) {
                    add_card(hand, card);
                }
            }
            if (hand.card_count == 0) {
                return 7;
            }
        }
        hands[hand_index] = hand;
    }
    if (hands[active_hand].finished) {
        return 8;
    }

    for (int index = 0; index < kOutcomeCount; ++index) {
        output[index] = 0;
    }
    for (std::uint64_t rollout = 0; rollout < rollouts; ++rollout) {
        SplitMix64 random(
            seed + 0x9e3779b97f4a7c15ULL * (rollout + 1)
        );
        const int profit = simulate_play_action(
            counts,
            total,
            unseen_unavailable,
            dealer_upcard,
            negative_peek != 0,
            hands,
            hand_count,
            active_hand,
            action_from_code(forced_action),
            random
        );
        if (profit < kMinimumProfitHalfUnits ||
            profit > kMaximumProfitHalfUnits) {
            return 9;
        }
        ++output[profit - kMinimumProfitHalfUnits];
    }
    return 0;
}
