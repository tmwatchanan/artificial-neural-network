function [ y ] = Sigmoid( v )
    y = 1.0 / (1.0 + exp(-v));
end

