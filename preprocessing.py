
# coding: utf-8

# In[1]:


f=open('your_data.fasta','r')
g=open('preprocssed_data.fasta','a') 
lines=f.readlines()
contex=3 
for line in lines:
    l=len(line)-1
    for i in range(0,(l-contex+1)):
        a=line[i:i+contex]
        x=str(a).replace("AAA","1").replace("TTT","2").replace("GAA","3").replace("AAG","4").replace("AAT","5").        replace("ATT","6").replace("CAA","7").replace("TGA","8").replace("TTC","9").replace("AGA","10").        replace("GAT","11").replace("AAC","12").replace("TAA","13").replace("TTA","14").replace("TCA","15").        replace("TAT","16").replace("ATG","17").replace("TGG","18").replace("ATC","19").replace("TTG","20").        replace("ATA","21").replace("GTT","22").replace("CTG","23").replace("CTT","24").replace("ACA","25").        replace("CAG","26").replace("CGA","27").replace("GGT","28").replace("GGC","29").replace("GCA","30").        replace("CAT","31").replace("GCG","32").replace("CGC","33").replace("GCT","34").replace("TCT","35").        replace("TCG","36").replace("ACC","37").replace("AGC","38").replace("CGG","39").replace("GAC","40").        replace("CCG","41").replace("CCA","42").replace("TGC","43").replace("ACG","44").replace("GGA","45").        replace("TGT","46").replace("ACT","47").replace("TAC","48").replace("AGT","49").replace("GCC","50").        replace("GAG","51").replace("GTA","52").replace("GTG","53").replace("AGG","54").replace("CGT","55").        replace("CAC","56").replace("GTC","57").replace("TCC","58").replace("CCT","59").replace("CTC","60").        replace("CTA","61").replace("GGG","62").replace("TAG","63").replace("CCC","64").replace(",\n","\n")
        if i<(1-contex+1):
            g.write(str(x)+",")
        else:
            g.write(str(x))
    g.write("\n")
f.close()
g.close()

