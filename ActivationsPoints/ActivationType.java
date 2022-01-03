package BordinNN.ActivationsPoints;

import BordinNN.ForwardI;

public enum ActivationType {
    RectifiedLinear(new ActivationReLU()),
    Softmax(new ActivationSoftMax()),
    Raw(new ActivationRaw());

    public final ActivationI activationI;

    ActivationType(ActivationI activationI){
        this.activationI = activationI;
    }
}
