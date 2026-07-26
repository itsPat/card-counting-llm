#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unordered_map>

namespace {

constexpr int kAce = 0;
constexpr int kTen = 9;
constexpr int kBust = 5;
constexpr int kBlackjack = 6;

using Counts = std::array<std::int16_t, 10>;
using Distribution = std::array<double, 7>;

struct State {
    Counts counts;
    std::int16_t hard_total;
    std::int8_t aces;

    bool operator==(const State&) const = default;
};

struct StateHash {
    std::size_t operator()(const State& state) const noexcept {
        std::size_t result = 1469598103934665603ULL;
        const auto mix = [&result](std::size_t value) {
            result ^= value;
            result *= 1099511628211ULL;
        };
        for (const auto count : state.counts) {
            mix(static_cast<std::size_t>(count));
        }
        mix(static_cast<std::size_t>(state.hard_total));
        mix(static_cast<std::size_t>(state.aces));
        return result;
    }
};

using Cache = std::unordered_map<State, Distribution, StateHash>;

struct CountsHash {
    std::size_t operator()(const Counts& counts) const noexcept {
        std::size_t result = 1469598103934665603ULL;
        for (const auto count : counts) {
            result ^= static_cast<std::size_t>(count);
            result *= 1099511628211ULL;
        }
        return result;
    }
};

int card_value(const int index) {
    return index == kAce ? 1 : index + 1;
}

int total_cards(const Counts& counts) {
    int total = 0;
    for (const auto count : counts) {
        total += count;
    }
    return total;
}

Distribution play_dealer(
    Counts& counts,
    const int hard_total,
    const int aces,
    const bool hit_soft_17,
    Cache& cache
) {
    const bool soft = aces > 0 && hard_total + 10 <= 21;
    const int total = soft ? hard_total + 10 : hard_total;
    Distribution result{};
    if (total > 21) {
        result[kBust] = 1.0;
        return result;
    }
    if (total > 17 || (total == 17 && !(soft && hit_soft_17))) {
        result[total - 17] = 1.0;
        return result;
    }

    const State state{counts, static_cast<std::int16_t>(hard_total),
                      static_cast<std::int8_t>(aces)};
    if (const auto found = cache.find(state); found != cache.end()) {
        return found->second;
    }

    const int cards = total_cards(counts);
    for (int index = 0; index < 10; ++index) {
        const int count = counts[index];
        if (count == 0) {
            continue;
        }
        --counts[index];
        const auto child = play_dealer(
            counts,
            hard_total + card_value(index),
            aces + (index == kAce ? 1 : 0),
            hit_soft_17,
            cache
        );
        ++counts[index];
        const double probability =
            static_cast<double>(count) / static_cast<double>(cards);
        for (int outcome = 0; outcome < 7; ++outcome) {
            result[outcome] += probability * child[outcome];
        }
    }
    cache.emplace(state, result);
    return result;
}

bool hole_allowed(
    const int hole,
    const int upcard,
    const bool no_blackjack
) {
    if (!no_blackjack) {
        return true;
    }
    if (upcard == kAce) {
        return hole != kTen;
    }
    if (upcard == kTen) {
        return hole != kAce;
    }
    return true;
}

Distribution compute_dealer_distribution(
    Counts& counts,
    const int upcard,
    const bool no_blackjack,
    const bool hit_soft_17,
    Cache& cache
) {
    int eligible = 0;
    for (int index = 0; index < 10; ++index) {
        if (hole_allowed(index, upcard, no_blackjack)) {
            eligible += counts[index];
        }
    }
    Distribution result{};
    if (eligible <= 0) {
        return result;
    }
    for (int hole = 0; hole < 10; ++hole) {
        const int count = counts[hole];
        if (count == 0 || !hole_allowed(hole, upcard, no_blackjack)) {
            continue;
        }
        const double probability =
            static_cast<double>(count) / static_cast<double>(eligible);
        if ((upcard == kAce && hole == kTen) ||
            (upcard == kTen && hole == kAce)) {
            result[kBlackjack] += probability;
            continue;
        }
        --counts[hole];
        const auto child = play_dealer(
            counts,
            card_value(upcard) + card_value(hole),
            (upcard == kAce ? 1 : 0) + (hole == kAce ? 1 : 0),
            hit_soft_17,
            cache
        );
        ++counts[hole];
        for (int outcome = 0; outcome < 7; ++outcome) {
            result[outcome] += probability * child[outcome];
        }
    }
    return result;
}

int hand_total(const std::int16_t* cards) {
    int hard = 0;
    for (int index = 0; index < 10; ++index) {
        hard += cards[index] * card_value(index);
    }
    return cards[kAce] > 0 && hard + 10 <= 21 ? hard + 10 : hard;
}

int profit_against_dealer(
    const int player_total,
    const int wager,
    const int dealer_outcome
) {
    if (player_total > 21) {
        return -wager;
    }
    if (dealer_outcome == kBlackjack) {
        return -wager;
    }
    if (dealer_outcome == kBust) {
        return wager;
    }
    const int dealer_total = dealer_outcome + 17;
    if (player_total > dealer_total) {
        return wager;
    }
    if (player_total == dealer_total) {
        return 0;
    }
    return -wager;
}

}  // namespace

extern "C" int blackjack_dealer_distribution(
    const int* raw_counts,
    const int upcard,
    const int no_blackjack,
    const int hit_soft_17,
    double* output
) {
    if (raw_counts == nullptr || output == nullptr || upcard < 0 || upcard > 9) {
        return 1;
    }
    Counts counts{};
    for (int index = 0; index < 10; ++index) {
        if (raw_counts[index] < 0 || raw_counts[index] > 32767) {
            return 2;
        }
        counts[index] = static_cast<std::int16_t>(raw_counts[index]);
    }

    Cache cache;
    cache.reserve(4096);
    const auto result = compute_dealer_distribution(
        counts,
        upcard,
        no_blackjack != 0,
        hit_soft_17 != 0,
        cache
    );
    double total = 0;
    for (const auto probability : result) {
        total += probability;
    }
    if (total == 0) {
        return 3;
    }
    std::memcpy(output, result.data(), sizeof(double) * result.size());
    return 0;
}

extern "C" int blackjack_split_distribution(
    const int* raw_counts,
    const int pair_card,
    const int upcard,
    const int no_blackjack,
    const int hit_soft_17,
    const int endpoint_count,
    const std::int16_t* endpoint_cards,
    const std::int16_t* endpoint_wagers,
    const std::uint64_t* endpoint_multiplicities,
    double* output
) {
    if (raw_counts == nullptr || endpoint_cards == nullptr ||
        endpoint_wagers == nullptr || endpoint_multiplicities == nullptr ||
        output == nullptr || pair_card < 0 || pair_card > 9 ||
        upcard < 0 || upcard > 9 || endpoint_count <= 0) {
        return 1;
    }
    Counts initial{};
    for (int index = 0; index < 10; ++index) {
        if (raw_counts[index] < 0 || raw_counts[index] > 32767) {
            return 2;
        }
        initial[index] = static_cast<std::int16_t>(raw_counts[index]);
    }

    Cache play_cache;
    play_cache.reserve(1 << 16);
    std::unordered_map<Counts, Distribution, CountsHash> dealer_cache;
    dealer_cache.reserve(1 << 15);

    for (int left = 0; left < endpoint_count; ++left) {
        const auto* left_cards = endpoint_cards + left * 10;
        const int left_total = hand_total(left_cards);
        const int left_wager = endpoint_wagers[left];
        for (int right = left; right < endpoint_count; ++right) {
            const auto* right_cards = endpoint_cards + right * 10;
            Counts remaining = initial;
            Counts drawn{};
            bool valid = true;
            for (int card = 0; card < 10; ++card) {
                drawn[card] = static_cast<std::int16_t>(
                    left_cards[card] + right_cards[card] -
                    (card == pair_card ? 2 : 0)
                );
                if (drawn[card] < 0 || drawn[card] > remaining[card]) {
                    valid = false;
                    break;
                }
            }
            if (!valid) {
                continue;
            }

            double probability = 1.0;
            for (int card = 0; card < 10 && valid; ++card) {
                for (int copy = 0; copy < drawn[card]; ++copy) {
                    const int total = total_cards(remaining);
                    int eligible = 0;
                    for (int hole = 0; hole < 10; ++hole) {
                        if (hole_allowed(hole, upcard, no_blackjack != 0)) {
                            eligible += remaining[hole];
                        }
                    }
                    const int count = remaining[card];
                    const int unavailable_as_hole =
                        hole_allowed(card, upcard, no_blackjack != 0) ?
                        count : 0;
                    const int numerator =
                        count * eligible - unavailable_as_hole;
                    if (eligible <= 0 || total < 2 || numerator <= 0) {
                        valid = false;
                        break;
                    }
                    probability *= static_cast<double>(numerator) /
                        static_cast<double>(eligible * (total - 1));
                    --remaining[card];
                }
            }
            if (!valid) {
                continue;
            }
            const std::uint64_t ordering = left == right ? 1 : 2;
            probability *= static_cast<double>(
                endpoint_multiplicities[left] *
                endpoint_multiplicities[right] * ordering
            );

            const int right_total = hand_total(right_cards);
            const int right_wager = endpoint_wagers[right];
            if (left_total > 21 && right_total > 21) {
                const int profit = -left_wager - right_wager;
                output[profit + 8] += probability;
                continue;
            }

            auto found = dealer_cache.find(remaining);
            if (found == dealer_cache.end()) {
                Counts dealer_counts = remaining;
                const auto dealer = compute_dealer_distribution(
                    dealer_counts,
                    upcard,
                    no_blackjack != 0,
                    hit_soft_17 != 0,
                    play_cache
                );
                found = dealer_cache.emplace(remaining, dealer).first;
            }
            for (int outcome = 0; outcome < 7; ++outcome) {
                if (found->second[outcome] == 0) {
                    continue;
                }
                const int profit =
                    profit_against_dealer(left_total, left_wager, outcome) +
                    profit_against_dealer(right_total, right_wager, outcome);
                output[profit + 8] +=
                    probability * found->second[outcome];
            }
        }
    }
    return 0;
}
