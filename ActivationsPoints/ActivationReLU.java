package BordinNN.ActivationsPoints;

import BordinNN.ForwardI;

import java.util.ArrayList;
import java.util.List;

public class ActivationReLU implements ActivationI {

    public List<List<Double>> output;
    public List<List<Double>> d;


    @Override
    public List<List<Double>> Forward(List<List<Double>> input) {
        output = Calculate(input);
        return output;
    }

    @Override
    public List<List<Double>> Backward(List<List<Double>> finalOutput) {
        List<List<Double>> output = new ArrayList<>();
        for (int i = 0; i < finalOutput.size(); i++) {
            List<Double> loss = new ArrayList<>();
            for (double value: finalOutput.get(i)) {
                if(value > 0 ? loss.add(1d) : loss.add(0d));
            }
            output.add(loss);
        }
        d=output;
        return output;
    }

    private List<List<Double>> Calculate(List<List<Double>> input){
        List<List<Double>> newOutput = new ArrayList<>();
        for (int i = 0; i < input.size(); i++) {
            List<Double> line = new ArrayList<>();
            for (double value : input.get(i)) {
                line.add(Math.max(0, value));
            }
            newOutput.add(line);
        }
        return newOutput;
    }
}
