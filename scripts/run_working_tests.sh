#!/bin/bash


STR=echo

shopt -s nullglob
files=(test_*py)

printf "%s\n" "${files[@]}"

echo $files

# counters for tests
win=0
fail=0

# list of failed test filenames
failures=()


for file in "${files[@]}"
do
   echo "running tests in $file"
   python $file working
   if [ $? -eq 0 ]
   then
     echo "test passed"
     ((win++))
   else
     echo "test failed"
     ((fail++))
     failures="${failures}, ${file}"

   fi
done

echo
echo
echo ----------------------------------------------------------------------
echo Succesful Tests:
echo $win
echo Failed Tests:
echo $fail

if [ $fail -ne 0 ]
    then
        echo -e "\n\nList of failed tests:"
        echo $failures
        echo -e "\n"
        echo ""
        exit 1
    else
        echo ""
        exit 0
    fi
