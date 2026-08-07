#!/usr/bin/env bash
# Rebuild the public single-commit release branch (h3-image-lora) from the
# current HEAD, excluding personal experiment files and patching train.sh to
# shareable defaults. Pure plumbing: no branch switching, no worktree changes.
#
#   .pyro/h3/release.sh          # rebuild branch locally + verify
#   PUSH=1 .pyro/h3/release.sh   # also force-push to the pyros-projects remote

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

BRANCH="${BRANCH:-h3-image-lora}"
REMOTE="${REMOTE:-pyros-projects}"

EXCLUDE=(
    docs/claude_review.md
    local
    .pyro/h3/release.sh
)
# every cfg except the lucy_v2 demo pair
while IFS= read -r cfg; do
    case "$cfg" in
        *lucy_v2*) ;;
        *) EXCLUDE+=("$cfg") ;;
    esac
done < <(git ls-tree -r --name-only HEAD .pyro/h3/cfg/)

export GIT_INDEX_FILE="$(mktemp)"
trap 'rm -f "$GIT_INDEX_FILE"' EXIT
git read-tree HEAD
git rm -r --cached -q --ignore-unmatch "${EXCLUDE[@]}"

# shareable launcher defaults for the release tree
blob=$(git show HEAD:.pyro/h3/train.sh \
    | sed -E \
        -e 's|^H3_NAME="\$\{H3_NAME:-[^}]*\}"|H3_NAME="${H3_NAME:-lucy_v2}"|' \
        -e 's|^SAMPLE_LORA_OVERLAY="\$\{SAMPLE_LORA_OVERLAY:-[^}]*\}"|SAMPLE_LORA_OVERLAY="${SAMPLE_LORA_OVERLAY:-}"|' \
        -e 's|^SAMPLE_LORA_OVERLAY_STRENGTH="\$\{SAMPLE_LORA_OVERLAY_STRENGTH:-[^}]*\}"|SAMPLE_LORA_OVERLAY_STRENGTH="${SAMPLE_LORA_OVERLAY_STRENGTH:-1.0}"|' \
    | git hash-object -w --stdin)
git update-index --cacheinfo "100755,$blob,.pyro/h3/train.sh"

tree=$(git write-tree)
commit=$(git commit-tree "$tree" -m "MiniMax H3 image LoRA trainer

Musubi Tuner fork with MiniMax H3 support: image-only LoRA training that
preserves video motion (no_packed_attn preset), real-video training with
joint audio, INT8 ConvRot model loading, image-tuned timestep schedules,
live training previews (stills or short videos, optional Turbo-LoRA
4-step preview overlays), and ComfyUI-format LoRA export. See the README
quick start and docs/minimax_h3.md.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>")
git branch -f "$BRANCH" "$commit"
unset GIT_INDEX_FILE

echo "== verify: leftovers that must NOT be in $BRANCH:"
if git ls-tree -r --name-only "$BRANCH" | grep -E "claude_review|^local/|release\.sh|cfg/(p_)?(cheststand|plushy|squishy|bb|bcatch|test_|woman|arched)"; then
    echo "RELEASE CONTAINS EXCLUDED FILES — aborting" >&2
    exit 1
fi
echo "   (none)"
echo "== verify: release train.sh defaults:"
git show "$BRANCH":.pyro/h3/train.sh | grep -E '^(H3_NAME|SAMPLE_LORA_OVERLAY|SAMPLE_LORA_OVERLAY_STRENGTH)='
echo "== files: $(git ls-tree -r --name-only "$BRANCH" | wc -l), commit: $(git rev-parse --short "$BRANCH")"

if [[ "${PUSH:-0}" == "1" ]]; then
    git push --force "$REMOTE" "$BRANCH"
    echo "pushed $BRANCH to $REMOTE"
fi
