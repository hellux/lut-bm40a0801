OBJDIR = build
ZIPNAME = mvdia_project
SRC = src/som.m src/feature_extraction_resnet50.m src/retraining.m

report:
	latexmk --outdir=${OBJDIR} -pdf doc/report.tex

zip: report
	rm -rf ${OBJDIR}/${ZIPNAME}
	mkdir ${OBJDIR}/${ZIPNAME}
	mkdir ${OBJDIR}/${ZIPNAME}/src
	${MAKE} report && cp ${OBJDIR}/report.pdf ${OBJDIR}/${ZIPNAME}
	cp ${SRC} ${OBJDIR}/${ZIPNAME}/src
	cp README.md matlab.sh ${OBJDIR}/${ZIPNAME}
	cd ${OBJDIR} && zip -r ${ZIPNAME}.zip ${ZIPNAME}
