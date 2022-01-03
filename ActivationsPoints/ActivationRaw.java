package BordinNN.ActivationsPoints;

import BordinNN.ForwardI;

import java.util.List;

public class ActivationRaw implements ActivationI{

    @Override
    public List<List<Double>> Forward(List<List<Double>> input) {
        return input;
    }

    @Override
    public List<List<Double>> Backward(List<List<Double>> finalOutput) {
        return finalOutput;
    }
}
