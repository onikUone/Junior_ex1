package eclipse_ex1;

import java.math.BigDecimal;
import java.util.Arrays;

public class InterNeuron {
	//member field
	BigDecimal[] weight;
	BigDecimal threshoud;
	BigDecimal net;
	BigDecimal eta;
	BigDecimal alpha;
	BigDecimal[] delta_W;
	BigDecimal delta_T;

	//member method
	public void clearNet(){
		net = new BigDecimal("0.0");
	}
	public void setNet(BigDecimal o_fromInput){
		net = net.add(weight[0].multiply(o_fromInput));
	}
	public void calcNet(){
		net = net.add(threshoud);
	}
	public BigDecimal output() {
		return sigmoid(net);
	}
	public static BigDecimal sigmoid(BigDecimal net) {
		double s = 1 / (1 + Math.exp((-1.0) * Double.parseDouble(String.valueOf(net))));
		return BigDecimal.valueOf(s);
	}
	public void reWeight(BigDecimal x, BigDecimal y, BigDecimal o_fromInter, BigDecimal o_fromOutput, OutputNeuron out){
		BigDecimal temp = new BigDecimal("0.0");
		for(int i=0; i<out.getWeight().length; i++){
			temp = temp.add(((((y.subtract(o_fromOutput))).multiply(o_fromOutput)).multiply((new BigDecimal("1.0").subtract(o_fromOutput)))).multiply(out.getWeight()[i]));
		}
		for(int i=0; i<weight.length; i++){
			delta_W[i] = delta_W[i].multiply(alpha);
			delta_W[i] = delta_W[i].add(eta.multiply(o_fromInter.multiply(((new BigDecimal("1.0")).subtract(o_fromInter)).multiply(temp.multiply(x)))));
			weight[i] = weight[i].add(delta_W[i]);
		}
		delta_T = delta_T.multiply(alpha);
		delta_T = delta_T.add(eta.multiply(o_fromInter.multiply((new BigDecimal("1.0").subtract(o_fromInter)).multiply(temp))));
		threshoud = threshoud.add(delta_T);
	}
	//constructor
	//引数に入力層の個数を指定すると、その個数だけの次元で結合強度配列を作成する。
	InterNeuron(int inputNumber) {
		weight = new BigDecimal[inputNumber];
		for(int i=0; i<inputNumber; i++){
			weight[i] = new BigDecimal("0.5");
		}
		threshoud = new BigDecimal("0.5");
		eta = new BigDecimal("0.5");
		alpha = new BigDecimal("0.9");
		delta_W = new BigDecimal[weight.length];
		Arrays.fill(delta_W, new BigDecimal("0.0"));
		delta_T = new BigDecimal("0.0");
	}
}
