package BordinNN.ActivationsPoints;

import java.util.List;

public class Input {
    public List<Double> values;
    public List<Double> trueResult;

    public Input(List<Double> values, List<Double> trueR){
        this.values = values;
        trueResult = trueR;
    }

    public void add(double value){
        values.add(value);
    }
}
