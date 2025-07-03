L=https://titan.gml-team.ru:5003/fsdownload/pvtSt8pW1/resnet34-b627a593.pth
wget --backups=1 -nv "$L" "$L"; rm "$(basename "$L").1"
L=https://titan.gml-team.ru:5003/fsdownload/wG3TPzLKZ/unique.pt
wget --backups=1 -nv "$L" "$L"; rm "$(basename "$L").1"
