"""
Filter AgentNet dataset for Word-related tasks on Windows using metadata.

Prerequisite: Run generate_mind2web_from_jsonl.py first to create the full mind2web JSON.

Uses meta_data_merged.jsonl for reliable app classification (domains, applications fields)
instead of regex matching on instruction/actual_task text.

Outputs:
1. word/word_tasks.jsonl         - Raw JSONL filtered for Word tasks
2. word/word_mind2web_style.json - Mind2Web-SoM format filtered for Word (1 screenshot = 1 sample)
3. word/word_review.txt          - Review file for manual verification

Usage:
    cd /home/thaole/thao_le/Magma
    python data-preprocessing/generate_mind2web_from_jsonl.py
    python data-preprocessing/filter_agentnet_word.py
"""

import json
import os

AGENTNET_JSONL = "/home/thaole/thao_le/Magma/datasets/agentnet/agentnet_win_mac_18k.jsonl"
METADATA_JSONL = "/home/thaole/thao_le/Magma/datasets/agentnet/meta_data_merged.jsonl"
MIND2WEB_JSON = "/home/thaole/thao_le/Magma/datasets/agentnet/agentnet_mind2web_style_full.json"
OUTPUT_DIR = "/home/thaole/thao_le/Magma/datasets/agentnet/word"

# Applications considered as Word processors
WORD_APPS = {"microsoft word", "libreoffice writer", "wps writer"}


def load_word_task_ids():
    """Load task_ids that have Word apps on Windows from metadata."""
    word_ids = {}  # task_id -> list of matched apps
    with open(METADATA_JSONL) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("system", "") != "Windows":
                continue
            apps = [a.lower() for a in entry.get("applications", [])]
            matched = [a for a in apps if a in WORD_APPS]
            if matched:
                word_ids[entry["task_id"]] = matched
    return word_ids


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Get Word task_ids from metadata
    print("Step 1: Loading Word task_ids from metadata...")
    word_ids = load_word_task_ids()
    print(f"  Word tasks in metadata: {len(word_ids)}")

    # Step 2: Filter JSONL using metadata task_ids
    print("\nStep 2: Filtering JSONL for Word tasks...")
    word_images = set()
    word_task_count = 0
    app_counts = {}

    jsonl_out = os.path.join(OUTPUT_DIR, "word_tasks.jsonl")
    review_out = os.path.join(OUTPUT_DIR, "word_review.txt")

    with (
        open(AGENTNET_JSONL) as f_in,
        open(jsonl_out, "w") as f_out,
        open(review_out, "w") as f_review,
    ):
        f_review.write(
            f"{'task_id':40s} | {'applications':30s} | stat | stp | instruction\n"
        )
        f_review.write("=" * 130 + "\n")

        for line in f_in:
            entry = json.loads(line)
            if entry["task_id"] not in word_ids:
                continue

            matched_apps = word_ids[entry["task_id"]]
            entry["matched_apps"] = matched_apps
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

            for step in entry["traj"]:
                word_images.add(step["image"])

            status = "done" if entry["task_completed"] else "fail"
            steps = len(entry["traj"])
            apps_str = ", ".join(matched_apps)
            f_review.write(
                f"{entry['task_id'][:40]:40s} | {apps_str:30s} | {status:4s} | {steps:3d} | "
                f"{entry['instruction'][:70]}\n"
            )

            word_task_count += 1
            for app in matched_apps:
                app_counts[app] = app_counts.get(app, 0) + 1

    print(f"  Word tasks in JSONL: {word_task_count}")
    for app, count in sorted(app_counts.items(), key=lambda x: -x[1]):
        print(f"    {app:25s}: {count}")

    # Step 3: Filter mind2web JSON by matching image filenames + image exists on disk
    print("\nStep 3: Filtering mind2web JSON for Word entries with images...")
    image_dir = os.path.join(os.path.dirname(AGENTNET_JSONL), "office_images")
    on_disk = set(os.listdir(image_dir))

    with open(MIND2WEB_JSON) as f:
        mind2web = json.load(f)

    word_mind2web = [
        entry for entry in mind2web
        if entry["image"] in word_images and entry["image"] in on_disk
    ]

    json_out = os.path.join(OUTPUT_DIR, "word_mind2web_style.json")
    with open(json_out, "w") as f:
        json.dump(word_mind2web, f, indent=2, ensure_ascii=False)

    skipped = sum(1 for e in mind2web if e["image"] in word_images and e["image"] not in on_disk)
    print(f"  Mind2web entries (with images):  {len(word_mind2web)}")
    print(f"  Skipped (missing images):        {skipped}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"  Word tasks (JSONL):              {word_task_count}")
    print(f"  Word samples (mind2web JSON):    {len(word_mind2web)}")
    print(f"  Skipped (no image on disk):      {skipped}")
    print(f"  Output directory:                {OUTPUT_DIR}/")
    print(f"  Review file:                     {review_out}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
