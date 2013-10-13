import java.text.NumberFormat;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Locale;
import java.text.FieldPosition;
import java.io.PrintWriter;
import java.io.BufferedReader;
import java.io.StreamTokenizer;

class Matrix{
    private double[][] A;
    private int m, n;
    
    //constructor
    public Matrix (int m, int n) {
        this.m = m;
        this.n = n;
        A = new double[m][n];
    }
    
    public Matrix (int m, int n, double num) {
        this.m = m;
        this.n = n;
        A = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = num;
            }
        }
    }
    
    
    public Matrix (double[][] A, int m, int n) {
        this.A = A;
        this.m = m;
        this.n = n;
    }
    
    public Matrix (double value[], int m) {
        this.m = m;
        n = (m != 0 ? value.length/m : 0);
        //if m != 0 is ture, n = value.length/m, else n = 0
        if (m * n != value.length) {
            throw new IllegalArgumentException("Array length must be a multiple of m.");
        }
        A = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = value[i + j * m];
            }
        }
    }
    
    //error detection
    public Matrix (double[][] A) {
        m = A.length;
        n = A[0].length;
        for (int i = 0; i < m; i++) {
            if (A[i].length != n) {
                throw new IllegalArgumentException("Row's length should be the same");
            }
        }
        this.A = A;
    }
    
    
    
    

}