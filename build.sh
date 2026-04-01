#!/bin/bash

NAME="fgds"
VERSION="1.0.0"
BUILDTAG="HEAD"
RELEASE=2
RPMBUILD_TMPDIR=$(pwd)/tmp-rpmbuild

function make_srpm() {
  	[ -d ${RPMBUILD_TMPDIR} ] && rm -rf ${RPMBUILD_TMPDIR}
  	mkdir -p ${RPMBUILD_TMPDIR}/{SOURCES,SPECS}
  	git archive --format=tar.gz --prefix=${NAME}-${VERSION}/ --output=${RPMBUILD_TMPDIR}/SOURCES/${NAME}-${VERSION}.tar.gz ${BUILDTAG}
  	sed -e "s/@VERSION@/${VERSION}/g" -e "s/@RELEASE@/${RELEASE}/g" fgds.spec.in > ${RPMBUILD_TMPDIR}/SPECS/fgds.spec
  	rpmbuild -bs --define "_topdir ${RPMBUILD_TMPDIR}" ${RPMBUILD_TMPDIR}/SPECS/fgds.spec
}

make_srpm