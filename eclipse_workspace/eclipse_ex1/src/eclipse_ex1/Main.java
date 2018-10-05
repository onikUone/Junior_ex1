package eclipse_ex1;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

public class Main {

	public static BigDecimal[][] readFile(String path) throws IOException{
		List<String[]> list = new ArrayList<String[]>();
		BufferedReader in = new BufferedReader(new FileReader(path));
		String line;
		while((line = in.readLine()) != null){
//			System.out.println(line);	//一行ずつprintできる
			list.add(line.split("\t"));
		}
		in.close();
		BigDecimal[][] x = new BigDecimal[list.size()][list.get(0).length];
		for(int i=0; i<list.size(); i++){
			x[i][0] = new BigDecimal(list.get(i)[0]);
			x[i][1] = new BigDecimal(list.get(i)[1]);
		}
		return x;
	}

	public static void writeFile(String path, BigDecimal x[], BigDecimal y[]) throws IOException{
		PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(path)));
		for(int i=0; i<x.length; i++){
			out.write(String.valueOf(x[i]));
			out.write("\t");
			out.write(String.valueOf(y[i]));
			out.write("\n");

		}
		out.close();
	}

	public static BigDecimal forward(BigDecimal x, InterNeuron inter[], OutputNeuron out, int flg){	//flg 出力層の出力表示フラグ
		for(int i=0; i<inter.length; i++){
			inter[i].clearNet();
		}
		out.clearNet();
		for(int i=0; i<inter.length; i++){
			inter[i].setNet(x);
		}
		for(int i=0; i<inter.length; i++){
			inter[i].calcNet();
		}
		for(int i=0; i<inter.length; i++){
			out.setNet(inter[i].output(), i);
		}
		out.calcNet();
		if(flg == 1){
			System.out.println(out.output());
		}
		return out.output();
	}

	public static void train(BigDecimal x[], BigDecimal y[], InterNeuron inter[], OutputNeuron out){
		for(int i=0; i<x.length; i++){
			//順方向計算
			BigDecimal o_fromOutput;	//計算量退避用
			BigDecimal o_fromInter[] = new BigDecimal[inter.length];	//計算量退避用
			//順方向1周
			forward(x[i], inter, out, 0);
			for(int j=0; j<inter.length; j++){
				o_fromInter[j] = inter[j].output();
			}
			o_fromOutput = out.output();

			//バックプロパゲーション
			out.reWeight(y[i], o_fromOutput, o_fromInter);
			for(int j=0; j<inter.length; j++){
				inter[j].reWeight(x[i], y[i], o_fromInter[j], o_fromOutput, out);
			}
		}
	}

	public static void main(String[] args) throws IOException {

		//BigDecimal test

		//BigDecimal test final

		int trainCount = 1;	//学習回数

		//ファイル読み込み
		BigDecimal[][] inputFile;	//datファイル ２次元配列化
		String readPath = "/Users/Uone/IDrive/OPU/研究フォルダ/1_プログラミング課題/eclipse_workspace/eclipse_ex1/src/eclipse_ex1/inputData.dat";	//Mac(ノートPC)環境
//		String readPath = "C:\\Users\\Yuichi Omozaki\\IDrive\\Junior_ex1\\eclipse_workspace\\eclipse_ex1\\src\\eclipse_ex1/inputData.dat";	//Windows(研究室環境)
		inputFile = readFile(readPath);

		//教師データ作成
		BigDecimal[] x = new BigDecimal[inputFile.length];	//入力ベクトル_教師データ
		BigDecimal[] y = new BigDecimal[inputFile.length];	//出力ベクトル_教師データ
		for(int i=0; i<inputFile.length; i++){
			x[i] = inputFile[i][0];
			y[i] = inputFile[i][1];
		}
		System.out.println("Train Data");
		System.out.println(" x   y ");
		for(int i=0; i<x.length; i++){
			System.out.println(x[i] + " " + y[i]);
		}
		System.out.println("main");

		//ニューロンオブジェクト
		InterNeuron inter[] = new InterNeuron[20];	//中間層ニューロンの個数はここで指定する
		for(int i=0; i<inter.length; i++){
			inter[i] = new InterNeuron(1);
		}
		OutputNeuron out = new OutputNeuron(inter.length);	//inter.lengthで中間層ニューロンの個数指定し、その数だけ結合強度を用意する。

		for(int i=0; i<trainCount; i++){	//ここで学習回数を指定できる
			train(x, y, inter, out);
		}
		System.out.println("train finished.");

		//学習関数出力
		BigDecimal[] test_X = new BigDecimal[100+1];
		BigDecimal[] test_Y = new BigDecimal[100+1];
		test_X[0] = new BigDecimal("0.0");
		test_Y[0] = forward(test_X[0], inter, out, 0);
		for(int i=1; i<test_X.length; i++){
			test_X[i] = test_X[i-1].add(new BigDecimal("0.01"));
			test_Y[i] = forward(test_X[i], inter, out, 0);
		}

		//ファイル書き出し
		String writePath = "/Users/Uone/IDrive/OPU/研究フォルダ/1_プログラミング課題/eclipse_workspace/eclipse_ex1/src/eclipse_ex1/outputData.dat";
		writeFile(writePath, test_X, test_Y);

		//評価関数
		BigDecimal e = new BigDecimal("0.0");
		for(int i=0; i< x.length; i++){
			e = e.add(((y[i].subtract(forward(x[i],inter,out,0))).multiply(y[i].subtract(forward(x[i],inter,out,0)))).divide(new BigDecimal("2.0"),6));
		}
		System.out.println("Count: " + trainCount);
		System.out.println("Error: " + e);
	}
}
