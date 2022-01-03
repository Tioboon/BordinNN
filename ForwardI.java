package BordinNN;

import BordinNN.*;

import java.util.List;

public interface ForwardI {

    List<Double> Forward(List<List<Double>> input);
    List<Double> Backward(List<List<Double>> finalOutput, int index);

}
