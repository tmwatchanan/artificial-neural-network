function [ localGradient ] = OutputLayerLocalGradient(error, y)
    localGradient = error * y * (1 - y);
end