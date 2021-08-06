# Copy from code to python files in build
# Run From src dir

for file in src/ports/postgres/modules/progressive/*; do
    in_file="$(basename "$file")"
    base_file="$(cut -d '_' -f1 <<<"$in_file")"
    echo $base_file
    echo $in_file
    cp src/ports/postgres/modules/progressive/"$in_file" build/src/ports/postgres/13/modules/progressive/"$base_file"
done
