#!/bin/bash
# test.sh for nwputhesis
# Run specific test or all tests

if [ $# -eq 0 ]; then
    l3build check;
    exit
fi

test_name="$1";
test_file="$1.tex";

file_path="testfiles/$test_file";
support_dir="testfiles/support";

if [ -f "$file_path" ]; then
    echo "Running test: $file_path";
    l3build check --file "$file_path";
else
    for test_dir in testfiles/*; do
        if [ -d "$test_dir" ]; then
            file_path="$test_dir/$test_file";
            if [ -f "$file_path" ]; then
                echo "Running test: $file_path";
                l3build check --file "$file_path";
                break
            fi
        fi
    done
fi