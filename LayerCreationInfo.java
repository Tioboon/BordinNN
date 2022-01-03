package BordinNN;

import BordinNN.ActivationsPoints.ActivationType;

public class LayerCreationInfo {
    public int numberOfNeurons;
    public ForwardI activationI;
    public ActivationType activationType;

    public LayerCreationInfo(int n, ActivationType name){
        numberOfNeurons = n;
        activationType = name;
    }
}
