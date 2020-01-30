changed_examples=`git diff --name-only master | grep -e examples/ | grep -e /**/*.py`
changed_sources=` git diff --name-only master | grep -v examples/ | grep -e /**/*.py`

for changed_example_ in $changed_examples
do
  python -m sphinx -W -D sphinx_gallery_conf.filename_pattern=$changed_example_ -b html -d _build/doctrees . _build/html
done

for changed_source_ in $changed_sources
do
  python -m sphinx -W -D sphinx_gallery_conf.filename_pattern=$changed_source_ -b html -d _build/doctrees . _build/html
done
