import sys
from mmsplice.vcf_dataloader import SplicingVCFDataloader
from mmsplice import MMSplice, predict_all_table
from mmsplice.utils import max_varEff

def evaluate_mmsplice(input_vcf, out_vcf, gtf, fasta):
    print(f'Evaluating MMSplice for {input_vcf}, output to {out_vcf}')
    dl = SplicingVCFDataloader(gtf, fasta, input_vcf, tissue_specific=True)
    model = MMSplice()
    predictions = predict_all_table(model, dl, pathogenicity=True, splicing_efficiency=True)
    predictionsMax = max_varEff(predictions)
    predictionsMax.to_csv(out_vcf, sep='\t', index=False)

def main():
    if len(sys.argv) < 5:
        print("Usage: python run_mmsplice.py <input_vcf> <out_vcf> <gtf> <fasta>")
        sys.exit(1)

    input_vcf = sys.argv[1]
    out_vcf = sys.argv[2]
    gtf = sys.argv[3]
    fasta = sys.argv[4]

    evaluate_mmsplice(input_vcf, out_vcf, gtf, fasta)
    print(f'MMSplice evaluation completed. Results saved to {out_vcf}')

if __name__ == "__main__":
    main()
