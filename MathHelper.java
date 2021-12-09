package BordinNN;

import java.util.Random;

public class MathHelper {

    public static double NonLinearF(double x){
        return 2 * Math.pow(x, 2);
    }

    public static double GetRandomPositiveOrNegative(double value){
        Random rnd = new Random();
        boolean bool = rnd.nextBoolean();
        if(bool) return Math.abs(value);
        return -Math.abs(value);
    }
}
