# Copy from python files in build to code
# Run From src dir

for file in build/src/ports/postgres/13/modules/progressive/*; do
    base_file="$(basename "$file")"
    in_file="$base_file"_in
    echo $in_file
    cp build/src/ports/postgres/13/modules/progressive/"$base_file" src/ports/postgres/modules/progressive/"$in_file"
done