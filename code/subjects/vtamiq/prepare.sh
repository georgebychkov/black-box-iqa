cp model.py src
dos2unix ./patches/vtamiq.patch
dos2unix ./src/run_config.py
dos2unix ./src/run_main.py
cd src && patch -Nlp1 < ../patches/vtamiq.patch
