package test.dr.evomodel.bigfasttree;

import dr.evolution.io.NewickImporter;
import dr.evolution.tree.NodeRef;
import dr.evomodel.bigfasttree.BigFastTreeModel;
import dr.evomodel.operators.ScaleNodeHeightOperator;
import dr.evomodel.operators.SubtreeLeapOperator;
import dr.evomodel.operators.UniformNodeHeightOperator;
import dr.evomodel.tree.TreeModel;
import dr.evomodelxml.operators.NodeHeightOperatorParser;
import dr.inference.operators.AdaptationMode;
import dr.math.MathUtils;
import junit.framework.TestCase;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Verifies the incremental store/restore in BigFastTreeModel via a self-oracle:
 * the full tree state (every height, every parent edge, the root) is captured via
 * public getters before and after each proposal; after accept/restore the live tree
 * must equal the expected snapshot. Exercises height moves AND SubtreeLeap topology
 * moves with random accept/reject.
 */
public class BigFastTreeModelStoreRestoreTest extends TestCase {

    private static String kingmanNewick(int n, long seed) {
        Random rng = new Random(seed);
        List<String> sub = new ArrayList<>();
        List<Double> h = new ArrayList<>();
        for (int i = 0; i < n; i++) { sub.add("t" + i); h.add(0.0); }
        double t = 0.0;
        int k = n;
        while (k > 1) {
            t += -Math.log(rng.nextDouble()) / (k * (k - 1) / 2.0);
            int i = rng.nextInt(k);
            int j; do { j = rng.nextInt(k); } while (j == i);
            if (i > j) { int tmp = i; i = j; j = tmp; }
            String merged = "(" + sub.get(i) + ":" + (t - h.get(i)) + "," + sub.get(j) + ":" + (t - h.get(j)) + ")";
            sub.remove(j); h.remove(j);
            sub.set(i, merged); h.set(i, t);
            k--;
        }
        return sub.get(0) + ";";
    }

    private static int[] parents(TreeModel t) {
        int[] p = new int[t.getNodeCount()];
        for (int i = 0; i < t.getNodeCount(); i++) {
            NodeRef par = t.getParent(t.getNode(i));
            p[i] = par == null ? -1 : par.getNumber();
        }
        return p;
    }

    private static double[] heights(TreeModel t) {
        double[] h = new double[t.getNodeCount()];
        for (int i = 0; i < t.getNodeCount(); i++) h[i] = t.getNodeHeight(t.getNode(i));
        return h;
    }

    public void testStoreRestoreContract() throws Exception {
        BigFastTreeModel.INCREMENTAL_STORE_RESTORE = true;
        int n = 300, iters = 30000;
        MathUtils.setSeed(17);
        TreeModel tree = new BigFastTreeModel(new NewickImporter(kingmanNewick(n, 17)).importTree(null));
        UniformNodeHeightOperator nh = new UniformNodeHeightOperator(tree, 1.0);
        ScaleNodeHeightOperator root = new ScaleNodeHeightOperator(tree, 1.0, 0.75,
                NodeHeightOperatorParser.OperatorType.SCALEROOT, AdaptationMode.ADAPTATION_OFF, 0.25);
        SubtreeLeapOperator leap = new SubtreeLeapOperator(tree, 1.0, 0.01,
                SubtreeLeapOperator.DistanceKernelType.NORMAL, AdaptationMode.ADAPTATION_OFF, 0.2);

        Random rng = new Random(17 * 7 + 3);
        int topoChanged = 0;
        boolean pass = true;
        for (int it = 0; it < iters && pass; it++) {
            boolean accept = rng.nextBoolean();
            int pick = rng.nextInt(10);
            int[] pBefore = parents(tree); double[] hBefore = heights(tree); int rootBefore = tree.getRoot().getNumber();
            tree.storeModelState();
            try {
                if (pick < 6) nh.doOperation();
                else if (pick < 8) root.doOperation();
                else leap.doOperation();
            } catch (Exception e) { /* internal reject */ }
            int[] pAfter = parents(tree); double[] hAfter = heights(tree); int rootAfter = tree.getRoot().getNumber();
            if (!Arrays.equals(pAfter, pBefore)) topoChanged++;

            int[] pExp; double[] hExp; int rootExp;
            if (accept) { tree.acceptModelState(); pExp = pAfter; hExp = hAfter; rootExp = rootAfter; }
            else { tree.restoreModelState(); pExp = pBefore; hExp = hBefore; rootExp = rootBefore; }

            if (!Arrays.equals(parents(tree), pExp) || !Arrays.equals(heights(tree), hExp)
                    || tree.getRoot().getNumber() != rootExp) {
                System.out.println("store/restore contract violated at iter " + it + " accept=" + accept + " pick=" + pick);
                pass = false;
            }
        }
        assertTrue("topology path not exercised", topoChanged > 100);
        assertTrue(pass);
    }
}
