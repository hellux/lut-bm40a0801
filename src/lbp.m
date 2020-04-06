function hst = lbp(I, cw)
    [h, w] = size(I);
    hst = [];
    for y = 0:ceil(h/cw)-1
        for x = 0:ceil(w/cw)-1
            x_range = 1+cw*x:min(cw*(x+1), w);
            y_range = 1+cw*y:min(cw*(y+1), h);
            hst_cell = lbp_cell(I(y_range, x_range));
            hst = [hst hst_cell];
        end
    end
end

function hst = lbp_cell(I)
    [h, w] = size(I);
    neigh = [0 0; 1 0; 2 0; 2 1; 2 2; 1 2; 0 2; 0 1];
    coeffs = 2.^[0:7];
    uniform = [0 1 2 3 4 6 7 8 12 14 15 16 24 28 30 31 32 48 56 60 62 63 64 ...
        96 112 120 124 126 127 128 129 131 135 143 159 191 192 193 195 199 ...
        207 223 224 225 227 231 239 240 241 243 247 248 249 251 252 253 254 ...
        255];
    
    hst = zeros(1, 59);
    for y = 1:h
        for x = 1:w
            center = I(y, x);
            number = 0;
            for i = 1:8
                nx = x + neigh(i, 1);
                ny = y + neigh(i, 2);
                if 0 < ny && ny <= h && 0 < nx && nx <= w
                    neigbour = I(ny, nx);
                    if neigbour >= center
                        number = number + coeffs(i);
                    end
                end
            end

            i = find(uniform == number);
            if length(i) == 0
                i = 59;
            end
            hst(1, i) = hst(1, i) + 1;
        end
    end
    hst = hst / sum(hst);
end
