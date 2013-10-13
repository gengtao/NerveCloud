//Code written by Gengtao Jia 
//This program trains Restricted Boltzmann Machine in which
// visible, binary, stochastic pixels are connected to hidden,
// binary, stochastic feature detectors using symmetrically weighted
// connections. Learning is done with 1-step Contrastive Divergence.   
// The program assumes that the following variables are set externally:
// maxepoch  -- maximum number of epochs
// numhid    -- number of hidden units 
// batchdata -- the data that is divided into batches (numcases numdims numbatches)
// restart   -- set to 1 if learning starts from beginning 

package rbm;
import java.util.Random;
import java.lang.Math;

public class rbm {
	private double epsilonw = 0.1; 
	private double epsilonvb = 0.1;
	private double espilonhb = 0.1;
	private double weightcost = 0.0002;
	private double initialmomentum = 0.5;
	private double finalmomentum = 0.9;
	private int numhid;
	
	public int numdims;
	public int numbatches;
	public int epoch;
	public int maxepoch;
	
	private Matrix hidbiases = new Matrix(1,numhid);
	private Matrix visbiases = new Matrix(1,numdims);
	private Matrix poshidprobs = new Matrix(1,numhid);
	private Matrix neghidprobs = new Matrix(1,numhid);
	private Matrix posprods = new Matrix(numdims,numhid);
	private Matrix negprods = new Matrix(numdims,numhid);
	private Matrix vishidinc = new Matrix(numdims,numhid);
	private Matrix hdibiasinc = new Matrix(1,numhid);
	private Matrix visbiasinc = new Matrix(1,numhid);
	private Matrix data = new Matrix(1,numdims);
	private Matrix vishid = new Matrix(numdims,numhid);
	
    //generate the initial random weights of each path	
	public void getvishid(){
		double[][] temp1 = new double[numdims][numhid];
		Random randomgenerator = new Random();
		for(int i = 0; i < numdims; i++){
			for(int j = 0; j < numhid; j++){
				temp1[i][j] = 0.1 * randomgenerator.nextGaussian();
				}
			}
		vishid = new Matrix(temp1);
	}
	//Start calculate the positive phase
	//calculate the cured value of h0
	
	public Matrix getposphase(){
		Matrix product_tmp1 = data.times(vishid);
		//(1 * numdims) * (numdims * numhid)
		product_tmp1.plusEquals(hidbiases);
		//data*vishid + hidbiases
		double [] product_tmp2 = product_tmp1.getRowPackedCopy();
		int i2 = 0;
		while( i2 < product_tmp2.length){
			double tmp = product_tmp2[i2];
			product_tmp2[i2] = 1/(1 - Math.exp(tmp));
			i2++;
		}
		poshidprobs = new Matrix(product_tmp2, numhid);
		Matrix data_t = data.transpose();
		posprods = data_t.times(poshidprobs);
		
	//end of the positive phase calculation, find the binary presentation of h0
		int i3 =0;
		double [] tmp1 = poshidprobs.getRowPackedCopy();
		Random randomgenerator = new Random();
		while (i3 < tmp1.length){
			if (tmp1[i3] > randomgenerator.nextDouble())
				tmp1[i3] = 1;
			else tmp1[i3] = 0;
			i3++;
			}
		Matrix poshidstates = new Matrix(tmp1,tmp1.length);
		return (poshidstates);
		}
	
	//start calculate the negative phase
	//calculate the curved value of v1,h1
	public void getnegphase(Matrix poshidstates){
		//find the vector of v1
		Matrix product_tmp1 = poshidstates.times(vishid.transpose());
		//(1 * numhid) * (numhid * numdims) = (1 * numdims)
		product_tmp1.plusEquals(visbiases);
		//poshidstates*vishid' + visbiases
		double [] tmp1 = product_tmp1.getRowPackedCopy();
		int i1 = 0;
		while( i1 < tmp1.length){
			double tmp = tmp1[i1];
			tmp1[i1] = 1/(1 - Math.exp(tmp));
			i1++;
		}
		Matrix negdata = new Matrix(tmp1, tmp1.length); //tmp1.length = numdims
		
		//find the vector of h1
		Matrix product_tmp2 = negdata.times(vishid);
		//(1 * numdims) * (numdims * numhid) = (1 * numhid)
		product_tmp2.plusEquals(hidbiases);
		double [] tmp2 = product_tmp2.getRowPackedCopy();
		int i2 = 0;
		while( i2 < tmp2.length){
			double tmp = tmp2[i2];
			tmp2[i2] = 1/(1 - Math.exp(tmp));
			i2++;
			}
		neghidprobs = new Matrix(tmp2, tmp2.length);
		negprods = negdata.transpose().times(neghidprobs);
		//(numdims * 1) *(1 * numhid) = (numdims * numhid)
		}
	
	//update the weights and biases
	public void update(int epoch){
		double momentum;
		if (epoch > 5)
			momentum = finalmomentum;
		else
			momentum = initialmomentum;
		//vishidinc = momentum*vishidinc + epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
		vishidinc.timesEquals(momentum);
	    Matrix temp1 = posprods.minus(negprods);
	    Matrix temp2 = vishid.times(weightcost);
	    temp1.minusEquals(temp2).timesEquals(epsilonw);
	    vishidinc.plusEquals(temp1);
	    //vishid = vishid + vishidinc;
	    vishid.plusEquals(vishidinc);
	    }
	
	public int run(int epoch, int epochmax){
        getvishid();
		for (int i = epoch; i < epochmax; i++ ){
			Matrix poshidstates = getposphase();
			getnegphase(poshidstates);
			update(epoch);
		}
		return (0);
		}
	}
		
	

	

	
		

