package eclipse_ex1;

public class Neuron {
	// member field
	double weight[], threshoud;


	// member method
	public void setValue(double w[], double t){
		weight = w;	//結合強度
		threshoud = t;	//しきい値
	}

	public double output(double input[]){
		double net = 0;
		for(int i=0; i<3; i++){
			net += weight[i]*input[i];
		}
		net += threshoud;
		return sigmoid(net);
	}

	public static double sigmoid(double net){
		return 1/(1 + Math.exp((-1)*net));
	}

	//constructor
	Neuron(){

	}
}
