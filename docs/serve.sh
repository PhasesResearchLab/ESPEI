# script to make it easier to test the documentation
# make the docs, serve them on a local HTML server, then cleanup after
make html && cd build/html && python3 -m http.server && cd ../.. && make clean
