#!/usr/bin/env bash
# Color definitions
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m' # No Color

trap 'printf "${YELLOW}\nDownload interrupted. You can resume by re-running the command.\n${NC}"; exit 1' INT

display_help() {
    cat << EOF
Usage:
  hfd <REPO_ID> [--include include_pattern1 include_pattern2 ...] [--exclude exclude_pattern1 exclude_pattern2 ...] [--hf_username username] [--hf_token token] [--tool aria2c|wget] [-x threads] [-j jobs] [--dataset] [--local-dir path] [--revision rev]

Description:
  Downloads a model or dataset from Hugging Face using the provided repo ID.

Arguments:
  REPO_ID         The Hugging Face repo ID (Required)
                  Format: 'org_name/repo_name' or legacy format (e.g., gpt2)
Options:
  include/exclude_pattern The patterns to match against file path, supports wildcard characters.
                  e.g., '--exclude *.safetensor *.md', '--include vae/*'.
  --include       (Optional) Patterns to include files for downloading (supports multiple patterns).
  --exclude       (Optional) Patterns to exclude files from downloading (supports multiple patterns).
  --hf_username   (Optional) Hugging Face username for authentication (not email).
  --hf_token      (Optional) Hugging Face token for authentication.
  --tool          (Optional) Download tool to use: aria2c (default) or wget.
  -x              (Optional) Number of download threads for aria2c (default: 4).
  -j              (Optional) Number of concurrent downloads for aria2c (default: 5).
  --dataset       (Optional) Flag to indicate downloading a dataset.
  --local-dir     (Optional) Directory path to store the downloaded data.
                             Defaults to the current directory with a subdirectory named 'repo_name'
                             if REPO_ID is is composed of 'org_name/repo_name'.
  --revision      (Optional) Model/Dataset revision to download (default: main).

Example:
  hfd gpt2
  hfd bigscience/bloom-560m --exclude *.safetensors
  hfd meta-llama/Llama-2-7b --hf_username myuser --hf_token mytoken -x 4
  hfd lavita/medical-qa-shared-task-v1-toy --dataset
  hfd bartowski/Phi-3.5-mini-instruct-exl2 --revision 5_0
EOF
    exit 1
}

[[ -z "$1" || "$1" =~ ^-h || "$1" =~ ^--help ]] && display_help

REPO_ID=$1
shift

# Default values
TOOL="aria2c"
THREADS=4
CONCURRENT=5
HF_ENDPOINT=${HF_ENDPOINT:-"https://huggingface.co"}
INCLUDE_PATTERNS=()
EXCLUDE_PATTERNS=()
REVISION="main"

validate_number() {
    [[ "$2" =~ ^[1-9][0-9]*$ && "$2" -le "$3" ]] || { printf "${RED}[Error] $1 must be 1-$3${NC}\n"; exit 1; }
}

# Argument parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        --include) shift; while [[ $# -gt 0 && ! ($1 =~ ^--) && ! ($1 =~ ^-[^-]) ]]; do INCLUDE_PATTERNS+=("$1"); shift; done ;;
        --exclude) shift; while [[ $# -gt 0 && ! ($1 =~ ^--) && ! ($1 =~ ^-[^-]) ]]; do EXCLUDE_PATTERNS+=("$1"); shift; done ;;
        --hf_username) HF_USERNAME="$2"; shift 2 ;;
        --hf_token) HF_TOKEN="$2"; shift 2 ;;
        --tool)
            case $2 in
                aria2c|wget)
                    TOOL="$2"
                    ;;
                *)
                    printf "%b[Error] Invalid tool. Use 'aria2c' or 'wget'.%b\n" "$RED" "$NC"
                    exit 1
                    ;;
            esac
            shift 2
            ;;
        -x) validate_number "threads (-x)" "$2" 10; THREADS="$2"; shift 2 ;;
        -j) validate_number "concurrent downloads (-j)" "$2" 10; CONCURRENT="$2"; shift 2 ;;
        --dataset) DATASET=1; shift ;;
        --local-dir) LOCAL_DIR="$2"; shift 2 ;;
        --revision) REVISION="$2"; shift 2 ;;
        *) display_help ;;
    esac
done

# Generate current command string
generate_command_string() {
    local cmd_string="REPO_ID=$REPO_ID"
    cmd_string+=" TOOL=$TOOL"
    cmd_string+=" INCLUDE_PATTERNS=${INCLUDE_PATTERNS[*]}"
    cmd_string+=" EXCLUDE_PATTERNS=${EXCLUDE_PATTERNS[*]}"
    cmd_string+=" DATASET=${DATASET:-0}"
    cmd_string+=" HF_USERNAME=${HF_USERNAME:-}"
    cmd_string+=" HF_TOKEN=${HF_TOKEN:-}"
    cmd_string+=" HF_TOKEN=${HF_ENDPOINT:-}"
    cmd_string+=" REVISION=$REVISION"
    echo "$cmd_string"
}

# Check if aria2, wget, curl are installed
check_command() {
    if ! command -v $1 &>/dev/null; then
        printf "%b%s is not installed. Please install it first.%b\n" "$RED" "$1" "$NC"
        exit 1
    fi
}

check_command curl; check_command "$TOOL"

LOCAL_DIR="${LOCAL_DIR:-${REPO_ID#*/}}"
mkdir -p "$LOCAL_DIR/.hfd"

if [[ "$DATASET" == 1 ]]; then
    METADATA_API_PATH="datasets/$REPO_ID"
    DOWNLOAD_API_PATH="datasets/$REPO_ID"
    CUT_DIRS=5
else
    METADATA_API_PATH="models/$REPO_ID"
    DOWNLOAD_API_PATH="$REPO_ID"
    CUT_DIRS=4
fi

# Modify API URL, construct based on revision
if [[ "$REVISION" != "main" ]]; then
    METADATA_API_PATH="$METADATA_API_PATH/revision/$REVISION"
fi
API_URL="$HF_ENDPOINT/api/$METADATA_API_PATH"

METADATA_FILE="$LOCAL_DIR/.hfd/repo_metadata.json"

# Fetch and save metadata
fetch_and_save_metadata() {
    status_code=$(curl -L -s -w "%{http_code}" -o "$METADATA_FILE" ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} "$API_URL")
    RESPONSE=$(cat "$METADATA_FILE")
    if [ "$status_code" -eq 200 ]; then
        printf "%s\n" "$RESPONSE"
    else
        printf "%b[Error] Failed to fetch metadata from $API_URL. HTTP status code: $status_code.%b\n$RESPONSE\n" "${RED}" "${NC}" >&2
        rm $METADATA_FILE
        exit 1
    fi
}

check_authentication() {
    local response="$1"
    if command -v jq &>/dev/null; then
        local gated
        gated=$(echo "$response" | jq -r '.gated // false')
        if [[ "$gated" != "false" && ( -z "$HF_TOKEN" || -z "$HF_USERNAME" ) ]]; then
            printf "${RED}The repository requires authentication, but --hf_username and --hf_token is not passed. Please get token from https://huggingface.co/settings/tokens.\nExiting.\n${NC}"
            exit 1
        fi
    else
        if echo "$response" | grep -q '"gated":[^f]' && [[ -z "$HF_TOKEN" || -z "$HF_USERNAME" ]]; then
            printf "${RED}The repository requires authentication, but --hf_username and --hf_token is not passed. Please get token from https://huggingface.co/settings/tokens.\nExiting.\n${NC}"
            exit 1
        fi
    fi
}

if [[ ! -f "$METADATA_FILE" ]]; then
    printf "%bFetching repo metadata...%b\n" "$YELLOW" "$NC"
    RESPONSE=$(fetch_and_save_metadata) || exit 1
    check_authentication "$RESPONSE"
else
    printf "%bUsing cached metadata: $METADATA_FILE%b\n" "$GREEN" "$NC"
    RESPONSE=$(cat "$METADATA_FILE")
    check_authentication "$RESPONSE"
fi

should_regenerate_filelist() {
    local command_file="$LOCAL_DIR/.hfd/last_download_command"
    local current_command=$(generate_command_string)
    
    # If file list doesn't exist, regenerate
    if [[ ! -f "$LOCAL_DIR/$fileslist_file" ]]; then
        echo "$current_command" > "$command_file"
        return 0
    fi
    
    # If command file doesn't exist, regenerate
    if [[ ! -f "$command_file" ]]; then
        echo "$current_command" > "$command_file"
        return 0
    fi
    
    # Compare current command with saved command
    local saved_command=$(cat "$command_file")
    if [[ "$current_command" != "$saved_command" ]]; then
        echo "$current_command" > "$command_file"
        return 0
    fi
    
    return 1
}

fileslist_file=".hfd/${TOOL}_urls.txt"

if should_regenerate_filelist; then
    # Remove existing file list if it exists
    [[ -f "$LOCAL_DIR/$fileslist_file" ]] && rm "$LOCAL_DIR/$fileslist_file"
    
    printf "%bGenerating file list...%b\n" "$YELLOW" "$NC"
    
    # Convert include and exclude patterns to regex
    INCLUDE_REGEX=""
    EXCLUDE_REGEX=""
    if ((${#INCLUDE_PATTERNS[@]})); then
        INCLUDE_REGEX=$(printf '%s\n' "${INCLUDE_PATTERNS[@]}" | sed 's/\./\\./g; s/\*/.*/g' | paste -sd '|' -)
    fi
    if ((${#EXCLUDE_PATTERNS[@]})); then
        EXCLUDE_REGEX=$(printf '%s\n' "${EXCLUDE_PATTERNS[@]}" | sed 's/\./\\./g; s/\*/.*/g' | paste -sd '|' -)
    fi

    # Check if jq is available
    if command -v jq &>/dev/null; then
        process_with_jq() {
            if [[ "$TOOL" == "aria2c" ]]; then
                printf "%s" "$RESPONSE" | jq -r \
                    --arg endpoint "$HF_ENDPOINT" \
                    --arg repo_id "$DOWNLOAD_API_PATH" \
                    --arg token "$HF_TOKEN" \
                    --arg include_regex "$INCLUDE_REGEX" \
                    --arg exclude_regex "$EXCLUDE_REGEX" \
                    --arg revision "$REVISION" \
                    '
                    .siblings[]
                    | select(
                        .rfilename != null
                        and ($include_regex == "" or (.rfilename | test($include_regex)))
                        and ($exclude_regex == "" or (.rfilename | test($exclude_regex) | not))
                      )
                    | [
                        ($endpoint + "/" + $repo_id + "/resolve/" + $revision + "/" + .rfilename),
                        " dir=" + (.rfilename | split("/")[:-1] | join("/")),
                        " out=" + (.rfilename | split("/")[-1]),
                        if $token != "" then " header=Authorization: Bearer " + $token else empty end,
                        ""
                      ]
                    | join("\n")
                    '
            else
                printf "%s" "$RESPONSE" | jq -r \
                    --arg endpoint "$HF_ENDPOINT" \
                    --arg repo_id "$DOWNLOAD_API_PATH" \
                    --arg include_regex "$INCLUDE_REGEX" \
                    --arg exclude_regex "$EXCLUDE_REGEX" \
                    --arg revision "$REVISION" \
                    '
                    .siblings[]
                    | select(
                        .rfilename != null
                        and ($include_regex == "" or (.rfilename | test($include_regex)))
                        and ($exclude_regex == "" or (.rfilename | test($exclude_regex) | not))
                      )
                    | ($endpoint + "/" + $repo_id + "/resolve/" + $revision + "/" + .rfilename)
                    '
            fi
        }
        result=$(process_with_jq)
        printf "%s\n" "$result" > "$LOCAL_DIR/$fileslist_file"
    else
        printf "%b[Warning] jq not installed, using grep/awk for metadata json parsing (slower). Consider installing jq for better parsing performance.%b\n" "$YELLOW" "$NC"
        process_with_grep_awk() {
            local include_pattern=""
            local exclude_pattern=""
            local output=""
            
            if ((${#INCLUDE_PATTERNS[@]})); then
                include_pattern=$(printf '%s\n' "${INCLUDE_PATTERNS[@]}" | sed 's/\./\\./g; s/\*/.*/g' | paste -sd '|' -)
            fi
            if ((${#EXCLUDE_PATTERNS[@]})); then
                exclude_pattern=$(printf '%s\n' "${EXCLUDE_PATTERNS[@]}" | sed 's/\./\\./g; s/\*/.*/g' | paste -sd '|' -)
            fi

            local files=$(printf '%s' "$RESPONSE" | grep -o '"rfilename":"[^"]*"' | awk -F'"' '{print $4}')
            
            if [[ -n "$include_pattern" ]]; then
                files=$(printf '%s\n' "$files" | grep -E "$include_pattern")
            fi
            if [[ -n "$exclude_pattern" ]]; then
                files=$(printf '%s\n' "$files" | grep -vE "$exclude_pattern")
            fi

            while IFS= read -r file; do
                if [[ -n "$file" ]]; then
                    if [[ "$TOOL" == "aria2c" ]]; then
                        output+="$HF_ENDPOINT/$DOWNLOAD_API_PATH/resolve/$REVISION/$file"$'\n'
                        output+=" dir=$(dirname "$file")"$'\n'
                        output+=" out=$(basename "$file")"$'\n'
                        [[ -n "$HF_TOKEN" ]] && output+=" header=Authorization: Bearer $HF_TOKEN"$'\n'
                        output+=$'\n'
                    else
                        output+="$HF_ENDPOINT/$DOWNLOAD_API_PATH/resolve/$REVISION/$file"$'\n'
                    fi
                fi
            done <<< "$files"

            printf '%s' "$output"
        }

        result=$(process_with_grep_awk)
        printf "%s\n" "$result" > "$LOCAL_DIR/$fileslist_file"
    fi
else
    printf "%bResume from file list: $LOCAL_DIR/$fileslist_file%b\n" "$GREEN" "$NC"
fi

# Perform download
printf "${YELLOW}Starting download with $TOOL to $LOCAL_DIR...\n${NC}"

cd "$LOCAL_DIR"
if [[ "$TOOL" == "aria2c" ]]; then
    aria2c --console-log-level=error --file-allocation=none -x "$THREADS" -j "$CONCURRENT" -s "$THREADS" -k 1M -c -i "$fileslist_file" --save-session="$fileslist_file"
elif [[ "$TOOL" == "wget" ]]; then
    wget -x -nH --cut-dirs="$CUT_DIRS" ${HF_TOKEN:+--header="Authorization: Bearer $HF_TOKEN"} --input-file="$fileslist_file" --continue
fi

if [[ $? -eq 0 ]]; then
    printf "${GREEN}Download completed successfully. Repo directory: $PWD\n${NC}"
else
    printf "${RED}Download encountered errors.\n${NC}"
    exit 1
fi
