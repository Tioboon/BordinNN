package BordinNN.ActivationsPoints;

import java.util.List;

public interface ActivationI {
    List<List<Double>> Forward(List<List<Double>> input);
    List<List<Double>> Backward(List<List<Double>> finalOutput);
}
