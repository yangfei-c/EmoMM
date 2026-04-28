from __future__ import annotations

from application.user_state import UserStateEstimator

from cfg_utils import fmt_pair, get_quadrant, load_queries, save_csv


def main():
    queries = load_queries()
    estimator = UserStateEstimator()
    rows = []

    for q in queries:
        text = q["text"]
        z_q = estimator.infer_z_q(text)
        rows.append(
            {
                "base_id": q["base_id"],
                "quadrant": q["quadrant"],
                "quadrant_name": q["quadrant_name"],
                "expected_va": q["expected_va"],
                "language": q["language"],
                "sample_id": q["sample_id"],
                "text": text,
                "predicted_z_q": fmt_pair(z_q),
                "z_q_v": float(z_q[0]),
                "z_q_a": float(z_q[1]),
                "predicted_quadrant": get_quadrant(z_q),
            }
        )

        print(q["base_id"], fmt_pair(z_q), get_quadrant(z_q))

    save_csv(rows, "va_results.csv")


if __name__ == "__main__":
    main()