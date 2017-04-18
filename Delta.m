function [ delta ] = Delta(learningRate, localGradient, x)
    delta = (+1) * learningRate * localGradient * x;
end

