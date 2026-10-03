"""Make a GNU Octave compatible copy of IB2d's ``IBM_Blackbox``.

Usage: python octave_compat.py <IBM_Blackbox> <output_dir>

Three MATLAB-only constructs in ``IBM_Driver.m`` are replaced; the numerics are
untouched:

1. ``ver('MATLAB')`` (release check for ``.geo_connect`` files) -> constant;
2. ``read_Vertex_Points``: MATLAB's ``textscan('%f %f')`` pads the 1-number
   header line with NaN, Octave does not -> read header and points separately;
3. ``read_Spring_Points``: same padding issue for the optional 5th column
   (spring non-linearity, NaN -> 1) -> parse line by line.
"""
import os
import re
import shutil
import sys


def patch(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    p = os.path.join(dst, "IBM_Driver.m")
    s = open(p).read()

    old1 = """test_ver = ver('MATLAB');
year_ver = test_ver.Release;
year = str2num(year_ver(3:6));
lett = year_ver(7);"""
    new1 = "year = 2024; lett = 'a'; % octave_compat"

    i = s.index("function [N,xLag,yLag] = read_Vertex_Points")
    old2 = "C = textscan(fileID,'%f %f','CollectOutput',1);"
    j = s.index(old2, i)
    s = s[:j] + ("N0 = fscanf(fileID,'%f',1); V0 = fscanf(fileID,'%f',[2 Inf])'; "
                 "C = {[N0 NaN; V0]}; % octave_compat") + s[j + len(old2):]

    old3 = re.compile(r"C = textscan\(fileID,'%f %f %f %f %f','CollectOutput',1\);\s*"
                      r"fclose\(fileID\);[^\n]*\n\s*spring_info = C\{1\};[^\n]*")
    new3 = """% octave_compat: parse line by line (MATLAB textscan pads the 5th column with NaN)
    fgetl(fileID); spring_info = [NaN NaN NaN NaN NaN];
    while true
        ln = fgetl(fileID); if ~ischar(ln), break; end
        v = sscanf(ln,'%f')'; if isempty(v), continue; end
        spring_info(end+1,:) = [v NaN(1,5-numel(v))];
    end
fclose(fileID);        %Close the data file."""
    if old1 not in s or not old3.search(s):
        raise RuntimeError("IB2d source changed, octave_compat.py needs an update")
    s = s.replace(old1, new1)
    s = old3.sub(lambda m: new3, s, count=1)
    open(p, "w").write(s)


if __name__ == "__main__":
    patch(sys.argv[1], sys.argv[2])
