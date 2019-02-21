PROGRAM arpstern

!##################################################################
!##################################################################
!######                                                      ######
!######                 PROGRAM ARPSTERN                     ######
!######                                                      ######
!######                     Developed by                     ######
!######     Center for Analysis and Prediction of Storms     ######
!######                University of Oklahoma                ######
!######                                                      ######
!##################################################################
!##################################################################
!
!
!-----------------------------------------------------------------------
!
!  PURPOSE:
!
!  A stand alone program created for initializing the ARPS model with
!  a terrain data base obtained from NCAR.  This program
!  will set up the horizontal grid according to the latitude and
!  longitude of a point provided by the user for the domain of the
!  model run.  There is a choice of three different map projections
!  at the time this code was prepared.  The terrain elevation data
!  consists of 30 second, 5 minute and 1 degree resolutions obtained
!  from NCAR data services. The structure of this program is as
!  follows:
!
!
!  1. Determine the model domain and set up the analysis grid.
!     (subroutine setgrid)
!
!  2. Read the user specified data sets and fill the 1 degree blocks
!     with the highest resolution data avaliable. (subroutine getter
!     and readter)
!
!  3. Apply a filter (Barnes, or other scheme) to the data.
!     (subroutine barnes)
!
!  4. Determine and plot the shape/response functions for the given
!     filter. (subroutine response)
!
!  5. Calculate and plot the RMS (root mean square) difference
!     fields.  (subroutine rmsdif)
!
!  Note: ARPSTERN is the driver for the ARPS terrain
!        initialization.
!
!  To compile and link the program:
!
!    make arpstern
!
!  To execute the program:
!
!    arpstern < arpstern.input >! arpstern.output
!
!  To create a tar file of all relevant files (except for terrain
!  data):
!
!    make arpster.tar
!
!-----------------------------------------------------------------------
!
!  AUTHOR: Dan Weber
!          1/12/94
!  Modification history:
!
!    M. Xue 2/7/94.
!    User specification of terrain data directory. More error
!    checking when opening/reading files.
!
!    D. Weber  12/15/94.
!    Included pre-set values of input data and print statements
!    of variables and parameters.
!
!    K. Brewster 1/30/95
!    Write file before doing any plotting.
!
!    D. Weber 9/8/95.
!    Revised the definition of tema and temb in GETTER to include the
!    absolute value.  This allows the proper setting of the
!    DO LOOP limits for use in analyzing southern hemispheric
!    data.
!
!    1999/07/16 (Gene Bassett)
!    Added patches for Great Lakes and other areas.
!
!
!    2000/11/28 (D. Weber)
!    Added dynamic allocation for expanded f90 capabilites.  This
!    involved setting the nx and ny in the arpstern.input file 
!    and computing the ntx and nty for the job.  Ntx and nty are
!    no longer needed as well as terrain.inc.
!
!-----------------------------------------------------------------------
!
!  Input read in from the input file:
!
!
!  INPUT:
!
!
!  COMMON BLOCK VARIABLES:
!
!    nx       number of grid points in the x direction.
!    ny       number of grid points in the y direction.
!
!    analtype Type of analysis for use in filtering the data
!             analtype = 1  Barnes scheme
!                      = 2  others (not yet implemented)
!
!    mapproj  Type of map projection used to setup the analysis grid.
!             mapproj  = 1  Polar Stereographic projection
!                      = 2  Mercator projection
!                      = 3  Lambert
!
!    itertype Type of terrain data desired,
!             itertype = 30 second
!                      =  5 minute
!                      =  1 degree
!    rmsopt   RMS difference control parameter,
!             rmsopt   =  0 rms difference and background field will
!                           not be calculated for a 1 pass analysis.
!                      =  1 rms difference and background field will
!                           be calculated for a 1 pass analysis.
!             NOTE: for multiple pass scheme, the background field
!                   must be calculated!!!!!!
!
!    comtype  Computer type the arpstern11.f program will be run on
!             comtype  =  1 for IBM and other machine capable of
!                           declaring 2 byte integers
!                      =  4 for CRAY machines and other machines
!                           in which the minimum integer declaration
!                           is integer *8
!
!    knot     Influence factor for the barnes scheme
!    gamma    Shape factor for the multiple pass Barnes analysis
!    ipass    Number of passes for the Barnes routine
!    trulat   Latitude at which the map projection is true (degree N)
!    trulon   Longitude at which the map projection is true
!             (degree E)
!    sclfct   Map scale factor (1/1000000=1000000=sclfct)
!    dx       Analysis grid spacing in the e-w direction (m)
!    dy       Analysis grid spacing in the n-s direction (m)
!    ctrlat   Latitude of the center of the analysis grid (deg. N)
!    ctrlon   Longitude of the center of the analysis grid (deg. E)
!    tol      Tolerance level for the weight function
!    wdn      Number of grid points representing the wavelength
!             associated
!             with the rdnot initial response.
!    rdnot    First pass or initial response for the wnd length wave.
!
!  LOCAL VARIABLES:
!
!    glab     Graphic label for the terrain plots
!    nlat     Latitude of the northern most extent of the analysis
!             grid (degrees n)
!    slat     Latitude of the southern most extent of the analysis
!             grid (degrees n)
!    elon     Longitude of the eastern edge of the analysis grid
!             (degrees E)
!    wlon     Longitude of the western edge of the analysis grid
!             (degrees E)
!
!    nbufx    Number of additional terrain data points outside the
!             analysis domain used on the calculation of the
!             boundaries. East-west
!    nbufy    Number of additional terrain data points outside the
!             analysis domain used on the calculation of the
!             boundaries. North-south
!    ntx      Number of terrain data points in the east-west
!             direction
!    nty      Number of terrain data points in the north-south
!             direction
!    pi       The number pi.
!    temak    Working knot used by the program (user specified or
!             calc. in setgrid)  (m**2)
!
!
!    h        Initial data array (m)
!
!
!  OUTPUT:
!
!    hback    Analyzed Background field at data points (m)
!    hterain  Analyzed terrain data array written out to
!             arpstern.dat. (m)
!
!
!  TEMPORARY ARRAYS:
!
!
!    tem1,tem2
!    tem3,tem4
!
!    tem11,tem12,tem13,tem14,tem15,tem16
!
!    tem21,tem22,tem23,tem24,tem25,tem26,tem27,tem28
!
!-----------------------------------------------------------------------
!
!-----------------------------------------------------------------------
!
!  Variable Declarations.
!
!-----------------------------------------------------------------------

  IMPLICIT NONE
!
!-----------------------------------------------------------------------
!
!  INPUT:
!
!-----------------------------------------------------------------------
!


!
!-----------------------------------------------------------------------
!
!  COMMON BLOCK VARIABLES:
!
!-----------------------------------------------------------------------
!
  INTEGER :: nx                ! number of grid points in the
                               ! x-direction
  INTEGER :: ny                ! number of grid points in the
                               ! y-direction
  INTEGER :: ndx               ! number of data points in the
                               ! x-direction for the desired terrain
                               ! data
  INTEGER :: ndy               ! number of grid points in the
                               ! y-direction for the desired terrain
                               ! data
  INTEGER :: analtype          ! Type of analysis technique
  INTEGER :: mapproj           ! Type of map projection
  INTEGER :: itertype          ! Type of terrain data
  INTEGER :: rmsopt            ! RMS difference control parameter
  INTEGER :: comtype           ! Computer type, =1 for ibm, =4 for
                               ! cray

  CHARACTER (LEN=80   ) :: tdatadir  ! Directory containing the terrain
                                     ! data
  CHARACTER (LEN=80    ) :: terndir  ! Directory containing the ascii
                                     ! terrain data
  INTEGER :: ltdir             ! Length of none-blank part of
                               ! tdatadir.

  REAL :: knot                 ! Influence parameter for the barnes
                               ! scheme
  REAL :: gamma                ! Shape factor for the multiple pass
                               ! barnes.
  INTEGER :: ipass             ! Number of passes for the Barnes
                               ! routine

  REAL :: trulat (2)           ! Latitude at which the map projection
                               ! is true (degees N)
  REAL :: trulon               ! Longitude at which the map
                               ! projection is true (degrees E)
  REAL :: sclfct               ! Map scale factor (m)

  REAL :: dx                   ! Analysis grid spacing in the e-w dir
  REAL :: dy                   ! Analysis grid spacing in the n-s dir
  REAL :: ctrlat               ! Latitude of the center analysis grid
  REAL :: ctrlon               ! Longitude of the center of analysis
                               ! grid
  REAL :: tol                  ! Tolerance level for the weight
                               ! function
  REAL :: wdn                  ! Wavelength in terms of gridpoints
                               ! for the rdnot response
  REAL :: rdnot                ! Initial response to wdn length wave.

!
!-----------------------------------------------------------------------
!
!  LOCAL VARIABLES:
!
!-----------------------------------------------------------------------
!

  CHARACTER (LEN=65) :: glab   ! Graphic label for the terrain plots
  REAL :: nlat                 ! Latitude of the northern most
                               ! extent of the analysis grid
                               ! (degrees N)
  REAL :: slat                 ! Latitude of the southern most
                               ! extent of the analysis grid
                               ! (degrees N)
  REAL :: elon                 ! Longitude of the eastern edge of
                               ! analysis grid (degrees E)
  REAL :: wlon                 ! Longitude of the western edge of
                               ! analysis grid (degrees E)
  INTEGER :: nbufx             ! Number of buffer points in e-w
  INTEGER :: nbufy             ! Number of buffer points in n-s
  INTEGER :: ntx               ! Number of terrain grid points e-w
  INTEGER :: nty               ! Number of terrain grid points n-s
  REAL :: pi                   ! The number pi
  REAL :: temak                ! Working knot  (m**2)

  REAL, ALLOCATABLE :: h (:,:) ! Initial terrain data array
                               ! for use in the analysis (m)

!
!-----------------------------------------------------------------------
!
!  OUTPUT:
!
!-----------------------------------------------------------------------
!

  REAL, ALLOCATABLE :: hback(:,:) ! Analyzed background field at data
                               ! points (m)
  REAL, ALLOCATABLE :: hterain(:,:) ! Analyzed terrain data array at
                               ! analysis points for use in model.(m)

!
!-----------------------------------------------------------------------
!
!  TEMPORARY ARRAYS:
!
!-----------------------------------------------------------------------
!

  REAL, ALLOCATABLE :: tem1 (:)
  REAL, ALLOCATABLE :: tem2 (:)

  INTEGER*2, ALLOCATABLE ::  tem3(:,:) ! Array for reading 1km, 5min, 1deg
                             ! data
  INTEGER*2, ALLOCATABLE ::  tem4(:,:) ! Array used to store terrain data

  REAL, ALLOCATABLE :: tem11(:,:)
  REAL, ALLOCATABLE :: tem12(:,:)
  REAL, ALLOCATABLE :: tem13(:,:)
  REAL, ALLOCATABLE :: tem14(:,:)
  INTEGER, ALLOCATABLE :: tem15(:,:)
  INTEGER, ALLOCATABLE :: tem16(:,:)

  REAL, ALLOCATABLE :: tem21(:,:)
  REAL, ALLOCATABLE :: tem22(:,:)
  REAL, ALLOCATABLE :: tem23(:,:)
  REAL, ALLOCATABLE :: tem24(:,:)
  INTEGER, ALLOCATABLE :: tem25(:,:)
  INTEGER, ALLOCATABLE :: tem26(:,:)
  REAL, ALLOCATABLE :: tem27(:,:)
  REAL, ALLOCATABLE :: tem28(:,:)

!-----------------------------------------------------------------------
!
!  Misc. local variables:
!
!-----------------------------------------------------------------------
!
  REAL    :: tema,temb,temc,temd,trulat1,trulat2,cint
  INTEGER :: i,j,ibgn,iend,jbgn,jend
  INTEGER :: ierr,istatus
  INTEGER :: idummy,imin,jmin,imax,jmax
  REAL    :: rdummy,hmin,hmax

  REAL :: lat, lon, terdx,terdy
  REAL :: fix_wisc, fix_bahamas
!
!-----------------------------------------------------------------------
!
!  Defining COMMON Blocks:
!
!-----------------------------------------------------------------------

  COMMON /terrainc/ analtype,mapproj,itertype,rmsopt,comtype,           &
                   tdatadir,ltdir
  COMMON /barnesc/ knot,gamma,ipass
  COMMON /projectc/ trulat,trulon,sclfct
  COMMON /gridc/ dx,dy,ctrlat,ctrlon,tol,wdn,rdnot

!-----------------------------------------------------------------------
!
!  Defining namelists:
!
!-----------------------------------------------------------------------

  NAMELIST /terraind/analtype,mapproj,itertype,rmsopt,comtype,          &
                     tdatadir,terndir
  NAMELIST /grid/ nx,ny
  NAMELIST /barnesd/ knot,gamma,ipass
  NAMELIST /mapprojd/trulat1,trulat2,trulon,sclfct
  NAMELIST /gridd/ dx,dy,ctrlat,ctrlon,tol,wdn,rdnot

!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
!
!  Beginning of executable code...
!
!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

!
!  Set the Namelist variables before they are read in.
!  This is done for the case of a read error in the namelist input
!  file.  Namelist variables will be set to values provided in the
!  example in section 9.3 of the ARPS User's Guide.
!
  nx       = 67
  ny       = 67
  analtype = 1
  mapproj  = 1
  itertype = 1
  rmsopt   = 1
  comtype  = 1
  tdatadir = 'arpstern.data'
  terndir  = 'arpstern.data'

  knot     = 0.0
  gamma    = 1.00
  ipass    = 4

  trulat1  = 40.70
  trulat2  = 0.0
  trulon   = 248.00
  sclfct   = 1.0

  dx       = 1000.0
  dy       = 1000.0
  ctrlat   = 40.70
  ctrlon   = 248.00
  tol      = 1.0E-7
  wdn      = 2.00
  rdnot    = 0.006

!-----------------------------------------------------------------------
!
!  Read and Print of Namelist variables and parameters.
!
!-----------------------------------------------------------------------
!
  READ(5,grid)
  WRITE(6,'(/5x,a,i4)') 'nx was:   ',nx
  WRITE(6,'(/5x,a,i4)') 'ny was:   ',ny

  READ(5,terraind)
  WRITE(6,'(/5x,a,i4)') 'Analysis type was:     ',analtype
  WRITE(6,'(/5x,a,i4)') 'Projection type was:   ',mapproj
  WRITE(6,'(/5x,a,i4)') 'Terrain data type was: ',itertype

  IF(itertype == 1)THEN
    ndx = 120
    ndy = 120
  ELSEIF(itertype == 2)THEN
    ndx = 12
    ndy = 12
  ELSEIF(itertype == 3)THEN
    ndx = 2
    ndy = 2
  END IF

  WRITE(6,'(/5x,a,i6)')'number of x data points per block was:',ndx
  WRITE(6,'(/5x,a,i6)')'number of y data points per block was:',ndy
  WRITE(6,'(/5x,a,i4)') 'Rmsopt was:            ',rmsopt
  WRITE(6,'(/5x,a,i4)') 'Computer type was:     ',comtype
  WRITE(6,'(/5x,a,a)') 'The source data directory was:  ',tdatadir
  WRITE(6,'(/5x,a,a)') 'The analyzed data directory was:',terndir

  READ(5,barnesd)
  WRITE(6,'(/5x,a,f6.3)') 'knot was:           ',knot
  WRITE(6,'(/5x,a,f6.3)') 'Gamma was:           ',gamma
  WRITE(6,'(/5x,a,i4)')   'Ipass was:           ',ipass

  READ(5,mapprojd)
  WRITE(6,'(/5x,a,f8.2)') 'trulat1 was:        ',trulat1
  WRITE(6,'(/5x,a,f8.2)') 'trulat2 was:        ',trulat2
  WRITE(6,'(/5x,a,f8.2)') 'trulon   was:        ',trulon
  WRITE(6,'(/5x,a,f12.1)') 'The scale factor was:',sclfct

  READ(5,gridd)
  WRITE(6,'(/5x,a,f9.2)') 'The x grid spacing was:  ',dx
  WRITE(6,'(/5x,a,f9.2)') 'The y grid spacing was:  ',dy
  WRITE(6,'(/5x,a,f8.2)') 'The center latitude was: ',ctrlat
  WRITE(6,'(/5x,a,f8.2)') 'The center longitude was:',ctrlon
  WRITE(6,'(/5x,a,f12.9)') 'The tolerance level was: ',tol
  WRITE(6,'(/5x,a,f6.3)') 'The base line wavelength was:',wdn
  WRITE(6,'(/5x,a,f6.3)') 'The base line response was:',rdnot

!  allocate some arrays associated with nx and ny, ndx and ndy.
  ALLOCATE(hterain(nx,ny),STAT=istatus)
  hterain = 0
  ALLOCATE(tem1(nx),STAT=istatus)
  tem1 = 0
  ALLOCATE(tem2(ny),STAT=istatus)
  tem2 = 0
  ALLOCATE(tem3(ndx,ndy),STAT=istatus)
  tem3 = 0
  ALLOCATE(tem4(ndx,ndy),STAT=istatus)
  tem4 = 0
  ALLOCATE(tem11(nx,ny),STAT=istatus)
  tem11 = 0
  ALLOCATE(tem12(nx,ny),STAT=istatus)
  tem12 = 0
  ALLOCATE(tem13(nx,ny),STAT=istatus)
  tem13 = 0
  ALLOCATE(tem14(nx,ny),STAT=istatus)
  tem14 = 0
  ALLOCATE(tem15(nx,ny),STAT=istatus)
  tem15 = 0
  ALLOCATE(tem16(nx,ny),STAT=istatus)
  tem16 = 0
!  initial allocation complete....
 
!
!-----------------------------------------------------------------------
!
!  Test to see if the comtype is suitable....
!
!-----------------------------------------------------------------------
!
  IF(comtype /= 1.AND.comtype /= 4)THEN
    PRINT *,'comtype in terrain.input is not set to IBM or CRAY ',      &
            'type. Please reset to 1 for IBM or 4 for CRAY'
    STOP
  END IF
!
!-----------------------------------------------------------------------
!
!  setting ctrlon in terms of degrees east
!
!-----------------------------------------------------------------------
!
  IF(ctrlon < 0)THEN
    ctrlon=360+ctrlon
  END IF
!
!-----------------------------------------------------------------------
!
!  test to see if reset value is within longitudinal limits..
!
!-----------------------------------------------------------------------
!
  IF(ctrlon < 0.OR.ctrlon > 360)THEN
    PRINT *,'ctrlon is incorrectly set in terrain.input, please ',      &
            'reset, use degrees east'
    STOP
  END IF
!
!-----------------------------------------------------------------------
!
!  test to see if the ctrlat is within limits...
!
!-----------------------------------------------------------------------
!

  IF(ctrlat < -90.OR.ctrlat > 90)THEN
    PRINT *,'ctrlat in terrain.input is too large or too small. ',      &
            'must be set between -90. and +90. degree north.'
    STOP
  END IF
!
!-----------------------------------------------------------------------
!
!  setting trulon in terms of degrees east
!
!-----------------------------------------------------------------------
!
  IF(trulon < 0)THEN
    trulon=360+trulon
  END IF
!
!-----------------------------------------------------------------------
!
!  test to see if reset value is within longitudinal limits..
!
!-----------------------------------------------------------------------
!
  IF(trulon < 0.OR.trulon > 360)THEN
    PRINT *,'trulon is incorrectly set in terrain.input, please ',      &
            'reset, use degrees east'
    STOP
  END IF
!
!-----------------------------------------------------------------------
!
!  test to see if trulat is within limits...
!
!-----------------------------------------------------------------------
!

  IF(trulat (1) < -90.OR.trulat (1) > 90)THEN
    PRINT *,'trulat (1) in terrain.input is too large or too ',         &
            'small. must be set between -90 and +90 degree north.'
    STOP
  END IF

  IF(trulat (2) < -90.OR.trulat (2) > 90)THEN
    PRINT *,'trulat (2) in terrain.input is too large or too ',         &
            'small. must be set between -90 and +90 degree north.'
    STOP
  END IF

  ltdir=LEN(tdatadir)
  CALL strlnth( tdatadir , ltdir)
  IF( ltdir  == 0 ) THEN
    tdatadir = '.'
    ltdir  =1
  END IF

  IF( tdatadir(ltdir:ltdir) /= '/') THEN
    ltdir=ltdir+1
    tdatadir(ltdir:ltdir)='/'
  END IF

  PRINT*,tdatadir(ltdir:ltdir)

  pi=4.0*ATAN(1.0)
  trulat (1)=trulat1
  trulat (2)=trulat2
  temak=knot
  CALL opngks

!
!-----------------------------------------------------------------------
!
!  Set the base grid parameters.
!
!-----------------------------------------------------------------------
!
  CALL setgrid(nx,ny,ndx,ndy,pi,                              &
           ntx,nty,nbufx,nbufy,slat,nlat,wlon,elon,temak,     &
           tema,temb,temc,temd,terdx,terdy,                   &
           tem1,tem2,tem11,tem12,tem13,tem14) 
!
!-----------------------------------------------------------------------
!
!  Allocate the space for the variables used in the analysis.  Can't
!  do this until the grid parameters are computed (above).
!
!-----------------------------------------------------------------------
!

  ALLOCATE(h(ntx,nty),STAT=istatus)
  h = 0
  ALLOCATE(hback(ntx,nty),STAT=istatus)
  hback = 0
  ALLOCATE(tem21(ntx,nty),STAT=istatus)
  tem21 = 0
  ALLOCATE(tem22(ntx,nty),STAT=istatus)
  tem22 = 0
  ALLOCATE(tem23(ntx,nty),STAT=istatus)
  tem23 = 0
  ALLOCATE(tem24(ntx,nty),STAT=istatus)
  tem24 = 0
  ALLOCATE(tem25(ntx,nty),STAT=istatus)
  tem25 = 0
  ALLOCATE(tem26(ntx,nty),STAT=istatus)
  tem26 = 0
  ALLOCATE(tem27(ntx,nty),STAT=istatus)
  tem27 = 0
  ALLOCATE(tem28(ntx,nty),STAT=istatus)
  tem28 = 0

!
!-----------------------------------------------------------------------
!
!  Set the grid arrays for the model and the extended buffer area.
!
!-----------------------------------------------------------------------
!

      call compgrid(nx,ny,ndx,ndy,pi,                         &
           ntx,nty,nbufx,nbufy,slat,nlat,wlon,elon,temak,     &
           tema,temb,temc,temd,terdx,terdy,                   &
           tem1,tem2,tem11,tem12,tem13,tem14,tem15,tem16,     &
           tem21,tem22,tem23,tem24,tem25,tem26)

!
!-----------------------------------------------------------------------
!
!  Call the GETTER subroutine to read the terrain data files and
!  fill the initial terrain array (h).
!
!-----------------------------------------------------------------------
!
  CALL getter(ntx,nty,ndx,ndy,slat,nlat,wlon,elon,                    &
              h,                                                        &
              tem3,tem4)
   print *,'initial terrain field'
   WRITE (*,'(10f7.1)') ((h(i,j),i=1,10),j=1,10)
!
!-----------------------------------------------------------------------
!
!  Fill in Great Lakes.  The original data set corresponds to the
!  bottom of the lakes.
!
!-----------------------------------------------------------------------
!
  CALL fix_lake_eliv(h,tem23,tem24,ntx,nty)
!
!-----------------------------------------------------------------------
!
!    Correct for bad terrain point in Wisc. & Bermuda.
!
!-----------------------------------------------------------------------
!
  DO j=1,nty
    DO i=1,ntx
      lat = tem23(i,j)
      lon = tem24(i,j) - 360.0

      fix_wisc = 290.0
      IF (lat > 43.0 .AND. lat < 44.0 .AND. lon > -90.5 .AND.           &
            lon < -89.0 .AND. h(i,j) > 500.0) THEN
        PRINT 9901, 'ARPSTERN: Lowering Wisc: i,j,0,lat,lon,h: ',       &
                       i,j,0,lat,lon,h(i,j),fix_wisc
        h(i,j) = fix_wisc
      END IF

      fix_bahamas = 0.0
      IF (lat >= 22.5 .AND. lat <= 24.0 .AND. lon >= -76.5 .AND.        &
            lon <= -75.0 .AND. ABS(h(i,j)) > 200.0) THEN
        PRINT 9901, 'ARPSTERN: Raising Bahamas: i,j,0,lat,lon,h: ',     &
                       i,j,0,lat,lon,h(i,j),fix_bahamas
        h(i,j) = fix_bahamas
      END IF

    END DO
  END DO

  9901 FORMAT (a,3I4,2F8.2,2F8.0)

!
!-----------------------------------------------------------------------
!
!    This section creates an analytical function for testing purposes
!
!-----------------------------------------------------------------------
!
!   DO 2 i=1,ntx
!
!  DO 2 j=1,nty
!   h(i,j)=20.*SIN(2.*pi*real((i-1.)/12.))  ! the denominator is the
                                  ! horizontal wavelength.....


! compute max and min of h...

   hmax = -1000000.
   hmin =  1000000.
   DO j=1,nty
   DO i=1,ntx
     IF(h(i,j) > hmax)THEN
       imax = i
       jmax = j
       hmax = h(i,j)
     END IF
     IF(h(i,j) < hmax)THEN
       hmin = h(i,j)
       imin = i
       jmin = j
     END IF
   END DO
   END DO
   print *,'hmax is =',hmax,imax,jmax
   print *,'hmin is =',hmin,imin,jmin
!
!-----------------------------------------------------------------------
!
!  Perform the desired analysis on the data base h.
!
!-----------------------------------------------------------------------

  IF (analtype == 1) THEN    ! perform a Barnes analysis on the data.

    DO i=1,ipass

      CALL barnes(i,1,nx,1,ny,nx,ny,ntx,nty,nbufx,nbufy,         &
                  temak,hback,h,                                 &
                  hterain,                                       &
                  tem11,tem12,tem21,tem22,tem15,tem16,tem13,tem14)


      IF(ipass > 1.OR.rmsopt == 1) THEN ! perform a barnes analysis
                                        ! at the data points

        ibgn=1+i*nbufx
        iend=ntx-i*nbufx
        jbgn=1+i*nbufy
        jend=nty-i*nbufy

        CALL barnes(i,ibgn,iend,jbgn,jend,ntx,nty,ntx,nty,         &
                    nbufx,nbufy,temak,hback,h,                     &
                    hback,                                         &
                    tem21,tem22,tem21,tem22,tem25,tem26,tem27,tem28)

      END IF

    END DO

!
!-----------------------------------------------------------------------
!
!  Write the analyzed terrain data (model grid data) into file
!  arpstern.dat.
!
!-----------------------------------------------------------------------
!
    CALL asnctl ('NEWLOCAL', 1, ierr)
    CALL asnfile('arpstern.dat', '-F f77 -N ieee', ierr)

    OPEN(11,FILE='arpstern.dat',FORM='unformatted',                     &
            STATUS='unknown')

    WRITE(11) nx,ny

    idummy = 0
    rdummy = 0.0

    WRITE(11) analtype,mapproj,itertype,ipass,idummy,                   &
              idummy,idummy,idummy,idummy,idummy,                       &
              idummy,idummy,idummy,idummy,idummy,                       &
              idummy,idummy,idummy,idummy,idummy

    WRITE(11) dx    ,dy    ,ctrlat,ctrlon,knot ,                        &
              gamma ,trulat1,trulat2,trulon,sclfct,                     &
              tol   ,wdn   ,rdnot ,rdummy,rdummy,                       &
              rdummy,rdummy,rdummy,rdummy,rdummy

    WRITE(11) hterain

    CLOSE(11)
!
!-----------------------------------------------------------------------
!
!  Plotting the results.
!
!-----------------------------------------------------------------------
!


!
!-----------------------------------------------------------------------
!
!  Plot the analysis according to the user defined map proj.
!
!-----------------------------------------------------------------------
!

    WRITE (glab,65) ipass,gamma
    65     FORMAT('Barnes Analysis (m) ','Pass=',i2,' Gamma=',f4.2)
    cint=0.
    CALL plot(nx,ny,nx,mapproj,trulat,trulon,cint,                      &
              tema,temb,temc,temd,glab,hterain)
!
!-----------------------------------------------------------------------
!
!  Plot the initial data field using a cylindrical equidistant proj.
!
!-----------------------------------------------------------------------
!

    PRINT *,'initial field, h'
    WRITE (*,'(10f7.1)') ((h(i,j),i=1,10),j=1,10)
    WRITE (glab,55) itertype
    55     FORMAT('Initial Terrain Field (m) ',' Itertype=',i2)
    cint=0.
    CALL plot(ntx,nty,ntx,8,trulat,trulon,cint,                        &
              tem23(1,nty),tem24(1,nty),tem23(ntx,1),                   &
              tem24(ntx,1),glab,h)
!
!-----------------------------------------------------------------------
!
!  Print the maximum initial data set value.
!
!-----------------------------------------------------------------------
!
    tema=-1000000.
    temb= 1000000.
    DO j=1,nty
      DO i=1,ntx
        tema=MAX(tema,h(i,j))
        temb=MIN(temb,h(i,j))
      END DO
    END DO
    PRINT *,'the maximum terrain initial value is',tema
    PRINT *,'the minimum terrain initial value is',temb

!
!-----------------------------------------------------------------------
!
!  Print the maximum analysis value.
!
!-----------------------------------------------------------------------
!

   hmax = -1000000.
   hmin =  1000000.
   DO j=1,ny
   DO i=1,nx
     IF(hterain(i,j) > hmax)THEN
       imax = i
       jmax = j
       hmax = hterain(i,j)
     END IF
     IF(hterain(i,j) < hmax)THEN
       hmin = hterain(i,j)
       imin = i
       jmin = j
     END IF
   END DO
   END DO
   print *,'hterain max is =',hmax,imax,jmax
   print *,'hterain min is =',hmin,imin,jmin
    tema=-100000.
    temb= 100000.
    DO j=1,ny
      DO i=1,nx
        tema=MAX(tema,hterain(i,j))
        temb=MIN(temb,hterain(i,j))
      END DO
    END DO
    PRINT *,'the maximum analysis value is',tema
    PRINT *,'the minimum analysis value is',temb

!
!-----------------------------------------------------------------------
!
!  Print the maximum background value.
!
!-----------------------------------------------------------------------
!
    IF(ipass > 1.OR.rmsopt == 1)THEN
      tema=-100.
      DO j=1,nty
        DO i=1,ntx
          tema=MAX(tema,hback(i,j))
        END DO
      END DO
      PRINT *,'the maximum background analysis value is',tema

!
!-----------------------------------------------------------------------
!
!  Plot the background data field using a cylindrical equidistant
!  proj.
!
!-----------------------------------------------------------------------
!

!
!-----------------------------------------------------------------------
!
!  Setting the lat/lon bounds for the plot routine.
!
!-----------------------------------------------------------------------
!
      ibgn= 1+ipass*nbufx
      iend= ntx-ipass*nbufx
      jbgn= 1+ipass*nbufy
      jend= nty-ipass*nbufy

      WRITE (glab,56) ipass,gamma
      56       FORMAT(1X,'Analyzed Terrain field (m)',                  &
               ' Pass=',i2,' Gamma=',f4.2)
      cint=0.
      CALL plot((iend-ibgn+1),(jend-jbgn+1),ntx,                       &
                8,trulat,trulon,cint,                                   &
                tem23(ibgn,jend),tem24(ibgn,jend),                      &
                tem23(iend,jbgn),tem24(iend,jbgn),                      &
                glab,hback(ibgn,jbgn))

!-----------------------------------------------------------------------
!
!  Call the RMS difference subroutine to compare the analysis of
!  data field at the data points to the initial data field.
!
!-----------------------------------------------------------------------

      PRINT *,'tem23,tem24 before rms'
      PRINT *,tem23(ibgn,jend),tem24(ibgn,jend),tem23(iend,jbgn)        &
             ,tem24(iend,jbgn)

      WRITE (glab,57) ipass,gamma
      57       FORMAT(1X,'RMS Difference at terrain points (m)',' Pass=' &
                      ,i2,' Gamma=',f4.2)

      CALL rmsdif(ibgn,iend,jbgn,jend,ntx,nty,                          &
                  tem23(ibgn,jend),tem24(ibgn,jend),                    &
                  tem23(iend,jbgn),                                     &
                  tem24(iend,jbgn),glab,h,hback,tem27)

    END IF

  ELSE IF(analtype == 2) THEN    !perform other type of analysis
    WRITE (6,'(a)') 'Other type of analysis is not implemented yet.'
!
!-----------------------------------------------------------------------
!
!  Other analysis methods have not yet been developed and implemented
!
!-----------------------------------------------------------------------
!

  END IF

  CALL clsgks

  STOP
END PROGRAM arpstern

!##################################################################
!##################################################################
!######                                                      ######
!######                SUBROUTINE SETGRID                    ######
!######                                                      ######
!######                     Developed by                     ######
!######     Center for Analysis and Prediction of Storms     ######
!######                University of Oklahoma                ######
!######                                                      ######
!##################################################################
!##################################################################
!
!

SUBROUTINE setgrid(nx,ny,ndx,ndy,pi,                          &
           ntx,nty,nbufx,nbufy,slat,nlat,wlon,elon,temak,     &
           temc,temd,teme,temf, terdx,terdy,                  &
           xg1d,yg1d,xg,yg,glat,glon)

!-----------------------------------------------------------------------
!
!  PURPOSE:
!
!  To set the grid of the analysis domain for use by the chosen
!  data analysis technique.
!
!  The structure of this program is as follows:
!
!    1. Read the arpstern.input file for the following required info:
!
!      - grid spacing in meters (dx,dy)
!
!      - type of map projection (polar stereographic, mercator,
!        lambert) and standard latitude(s) and longitude of the these
!        projections.
!
!      - scale factor of the projection (eg. 1/1000000=1000000)
!
!      - central latitude and longitude of the model grid
!
!-----------------------------------------------------------------------
!
!  AUTHOR: Dan Weber
!  1/12/94
!
!
!  REVISED: Dan Weber
!  6/17/94
!
!  Modified initial position to center of analysis grid.
!  Code cleanup and added comtype option for computer specific
!  record length requirements.
!
!  MODIFIED: Dan Weber
!  11/28/2000
! 
!  Removed computation of grid arrays for f90 allocation purposes.
!
!-----------------------------------------------------------------------
!
!
!
!  INPUT :
!
!    nx       Number of grid points for the analysis grid e-w dir.
!    ny       Number of grid points for the analysis grid n-s dir.
!    ndx      Number of data points in a 1 degree block (e-w)
!    ndy      Number of data points in a 1 degree block (n-s)
!    pi       The constant Pi
!
!-----------------------------------------------------------------------
!
!  COMMON BLOCK VARIABLES:
!
!-----------------------------------------------------------------------
!
!    analtype Analysis type (1=barnes,2=other,...)
!    mapproj  Type of map projection used to setup the analysis grid.
!             mapproj   = 1  Polar Stereographic projection
!                       = 2  Mercator projection
!                       = 3  Lambert
!    itertype Type of terrain desired (1=30 second, 2= 5 minute,
!                                      3= 1 degree)
!    rmsopt   Option for calc. the rms differences. (0=no,1=yes)
!    comtype  Computer type the arpstern11.f program will be run on
!             comtype  =  1 for IBM and other machine capable of
!                           declaring 2 byte integers
!                      =  4 for CRAY machines and other machines
!                           in which the minimum integer declaration
!                           is integer *8
!    knot     Shape parameter for the barnes weight function.
!    gamma    Shape factor for multi pass scheme
!    ipass    Total number of passes thru the barnes scheme
!    trulat   Latitude(s) at which the map projection is true
!             (degree N)
!    trulon   Longitude at which the map projection is true
!             (degree E)
!    sclfct   Map scale factor (m)
!
!    dx       Analysis grid spacing in the x-direction east-west (m)
!    dy       Analysis grid spacing in the y-direction north-south(m)
!    ctrlat   Latitude of the center of the analysis grid (deg. N)
!    ctrlon   Longitude of the center of the analysis grid (deg. E)
!    tol      Tolerance for the weight function used in determining
!             the cut-off radius or radius of influence
!    wdn      Wavelength in terms of grid points in which rdnot is
!             valid
!    rdnot    Initial Response for wdn length wave
!
!  LOCAL VARIABLES:
!
!    dincx    Analysis x-direction grid spacing normalized by the map
!             scale dincx=dx/sclfct
!    dincy    Analysis y-direction grid spacing normalized by the map
!             scale dincy=dy/sclfct
!    mterdx   Max. distance between e-w data pts. (m)
!    terdx    Distance in meters between data points in the e-w dir.
!             (m)
!    terdy    Distance in meters between data points in the n-s dir.
!             (m)
!
!  OUTPUT:
!
!    ntx      Number of data points needed in the east-west direction
!    nty      Number of data points needed in the north-south
!             direction
!    nbufx    Number of additional points needed to perform the
!             Barnes
!             analysis at the edges of the domain. e-w
!    nbufy    Number of additional points needed to perform the
!             Barnes
!             analysis at the edges of the domain. n-s
!    nlat     Latitude of the northern edge of the data area
!             (degrees N)
!    slat     Latitude of the southern edge of the data area
!             (degrees N)
!    elon     Longitude of the eastern edge of the data area
!             (degrees E)
!    wlon     Longitude of the western edge of the data area
!             (degrees E)
!    temak    Working value of knot (user or program specified)
!    xg1d     Analysis grid points e-w (1-d) in grid units
!             (=meters/sclfct)
!    yg1d     Analysis grid points n-s (1-d) in grid units
!             (=meters/sclfct)
!    xg       Analysis grid points in the e-w direction
!             (in grid units)
!    yg       Analysis grid points in the n-s direction
!             (in grid units)
!    glat     Latitude of the analysis data points
!    glon     Longitude of the analysis data points
!
!
!-----------------------------------------------------------------------
!

!
!-----------------------------------------------------------------------
!
!  Variable Declarations.
!
!-----------------------------------------------------------------------

  IMPLICIT NONE
!
!-----------------------------------------------------------------------
!
!  INPUT:
!
!-----------------------------------------------------------------------
!

  INTEGER :: nx                ! Number of analysis grid points e-w
  INTEGER :: ny                ! Number of analysis grid points n-s
  INTEGER :: ndx               ! Number of data points (e-w)
  INTEGER :: ndy               ! Number of data points (n-s)
  REAL :: pi                   ! The constant Pi

!
!-----------------------------------------------------------------------
!
!  COMMON BLOCK VARIABLES:
!
!-----------------------------------------------------------------------
!

  INTEGER :: analtype          ! Analysis type
  INTEGER :: mapproj           ! Type of map projection used in
                               ! analysis
  INTEGER :: itertype          ! Desired terrain resolution for
                               ! analysis
  INTEGER :: rmsopt            ! Option for enabling the rms diff.
                               ! routine.
  INTEGER :: comtype           ! Computer type, =1 for ibm, =4 for
                               ! cray

  CHARACTER (LEN=80   ) :: tdatadir  ! Directory containing the terrain
                                     ! data
  INTEGER :: ltdir             ! Length of none-blank part of
                               ! tdatadir.

  REAL :: knot                 ! Influence parameter for the barnes
                               ! scheme
  REAL :: gamma                ! Barnes shape factor for multi pass
                               ! scheme
  INTEGER :: ipass             ! Total number of passes


  REAL :: trulat (2)           ! Latitude of true map projection
                               ! (deg N)
  REAL :: trulon               ! Longitude of true map projection
                               ! (deg E)
  REAL :: sclfct               ! Map scale factor (m)

  REAL :: dx                   ! Analysis grid spacing e-w direction
                               ! (m)
  REAL :: dy                   ! Analysis grid spacing n-s direction
                               ! (m)
  REAL :: ctrlat               ! Latitude of the center of the
                               ! analysis grid (deg. n)
  REAL :: ctrlon               ! Longitude of the center of the
                               ! analysis grid (deg. e)
  REAL :: tol                  ! Tolerance level for the weight
                               ! function
  REAL :: wdn                  ! Wavelength in terms of analysis grid
                               ! points for the rdnot response
  REAL :: rdnot                ! Initial Response for wdn length wave

!
!-----------------------------------------------------------------------
!
!  LOCAL VARIABLES:
!
!-----------------------------------------------------------------------
!

  REAL :: dincx                ! Analysis x-direction grid
                               ! spacing/map scale
  REAL :: dincy                ! Analysis y-direction grid
                               ! spacing/map scale
  REAL :: mterdx               ! Max. distance between e-w data pts
  REAL :: terdx                ! Distance in grid meters between e-w
                               ! data pts
  REAL :: terdy                ! Distance in grid meters between n-s
                               ! data pts

!
!-----------------------------------------------------------------------
!
!  OUTPUT:
!
!-----------------------------------------------------------------------
!

  INTEGER :: ntx               ! Actual number of data points needed
                               ! for data domain e-w
  INTEGER :: nty               ! Actual number of data points needed
                               ! for data domain n-s
  INTEGER :: nbufx             ! Number of buffer points in the x
                               ! direction
  INTEGER :: nbufy             ! Number of buffer points in the y
                               ! direction
  REAL :: nlat                 ! Max. Latitude of the data area
                               ! (degrees n)
  REAL :: slat                 ! Min. Latitude of the data area
                               ! (degrees n)
  REAL :: elon                 ! Max. Longitude of the data area
                               ! (degrees e)
  REAL :: wlon                 ! Min. Longitude of the data area
                               ! (degrees e)
  REAL :: temak                ! Working value of knot
  REAL :: xg1d(nx)             ! Analysis grid points e-w (1-d) in
                               ! grid units
  REAL :: yg1d(ny)             ! Analysis grid points n-s (1-d) in
                               ! grid units
  REAL :: xg(nx,ny)            ! 2-D analysis grid points east-west
  REAL :: yg(nx,ny)            ! 2-D analysis grid points north-south
  REAL :: glat(nx,ny)          ! Latitude of the analysis data points
  REAL :: glon(nx,ny)          ! Longitude of the analysis data
                               ! points

!
!-----------------------------------------------------------------------
!
!  Misc. local variables:
!
!-----------------------------------------------------------------------
!
  INTEGER :: i,j,k,l,m,n,itema,itemb
  REAL :: tema,temb,temc,temd,teme,temf
!
!-----------------------------------------------------------------------
!
!  Defining COMMON blocks:
!
!-----------------------------------------------------------------------

  COMMON /terrainc/ analtype,mapproj,itertype,rmsopt,comtype            &
                    ,tdatadir,ltdir
  COMMON /projectc/ trulat,trulon,sclfct
  COMMON /gridc/ dx,dy,ctrlat,ctrlon,tol,wdn,rdnot
  COMMON /barnesc/ knot,gamma,ipass

!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
!
!  Beginning of executable code...
!
!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

  sclfct=1.0/sclfct
  dincx=dx*sclfct
  dincy=dy*sclfct

  PRINT *,'dincx,dincy'
  PRINT *,dincx,dincy
!
!-----------------------------------------------------------------------
!
!  Note IMPORTANT!!!!: dx and dy are in meters...and the grid is
!  oriented so that the y-axis runs along a longitude line towards
!  the northpole and the x-axis is perpendicular to the y-axis.
!  Create the x,y grid in grid meters (multiplied by sclfct), the
!  origin is the southwest corner of the analysis domain and is
!  translated from the center point specified by the user (lat,lon).
!
!
!-----------------------------------------------------------------------

  CALL setmapr(mapproj,sclfct,trulat,trulon)

  CALL lltoxy(1,1,ctrlat,ctrlon,tema,temb)
!
!-----------------------------------------------------------------------
!
!  translate the center point to the first physical point.
!
!-----------------------------------------------------------------------
!
  tema=tema-((nx-3)*dincx)/2
  temb=temb-((ny-3)*dincy)/2
!
!-----------------------------------------------------------------------
!
!  translating from the first physical point to the first scalar
!  point.
!
!-----------------------------------------------------------------------
!
  tema=tema-dincx/2.0
  temb=temb-dincy/2.0

!
!-----------------------------------------------------------------------
!
!  Calculate the rest of the analysis grid points in earth
!  meters*sclfct
!
!-----------------------------------------------------------------------
!

  DO j=1,ny
    yg1d(j)=temb+(j-1)*dincy
  END DO

  DO i=1,nx
    xg1d(i)=tema+(i-1)*dincx
  END DO

!
!-----------------------------------------------------------------------
!
!  Print a sample of the analysis grid values...
!
!-----------------------------------------------------------------------
!

  PRINT *,'xg'
  WRITE(*,'(6f12.1)') (xg1d(i),i=1,6)
  PRINT *,'yg'
  WRITE(*,'(3f12.1)') (yg1d(j),j=1,3)

!-----------------------------------------------------------------------
!
!  Need to determine the boundaries of the analysis grid domain in
!  terms of latitude and longitude.  This will be done by taking the
!  southwest in terms of analysis grid values (in meters) and
!  calculating the lat/lon.
!
!-----------------------------------------------------------------------

  CALL xytoll(nx,ny,xg1d,yg1d,glat,glon)
!
!-----------------------------------------------------------------------
!
!  Print a sample of the analysis grid lat/lons...
!
!-----------------------------------------------------------------------
!

  PRINT *,'glat'
  WRITE (*,'(10f7.2)') ((glat(i,j),i=1,10),j=1,3)
  PRINT *,'glon'
  WRITE (*,'(10f7.2)') ((glon(i,j),i=1,10),j=1,3)

!
!-----------------------------------------------------------------------
!
!   Converting the longitudes obtained from the xytoll program to
!   degrees east for use in the data sets.
!
!-----------------------------------------------------------------------
!
! commented out the follwoinig code testing 11/28/00
! DO j=1,ny
!   DO i=1,nx
!     IF(glon(i,j) < 0.0)glon(i,j)=glon(i,j)+360.
!  start of new code
!     IF(glon(i,j) >= 360.0)glon(i,j)=glon(i,j)-360.
!  end of new code
!   END DO
! END DO

! PRINT *,'glon adjusted'
! WRITE (*,'(10f7.2)') ((glon(i,j),i=1,nx),j=1,3)

!-----------------------------------------------------------------------
!
!  Calculating the number of data points to be included in the
!  buffered areas (nbufx,nbufy).  But first we need to determine the
!  smallest horizontal spacing of the user requested data set for use
!  in calculating nbufx and nbufy.
!
!  note: lat and lon are in degrees north and east...
!
!-----------------------------------------------------------------------

  nlat=-91.
  slat=91.
  elon=-900.
  wlon=900.

  DO i=1,nx
    nlat=MAX(glat(i,ny),nlat)
    slat=MIN(glat(i,1),slat)
    elon=MAX(glon(i,ny),elon)
    wlon=MIN(glon(i,1),wlon)
  END DO

  DO j=1,ny
    nlat=MAX(glat(1,j),nlat)
    nlat=MAX(glat(nx,j),nlat)
    slat=MIN(glat(1,j),slat)
    slat=MIN(glat(nx,j),slat)
    elon=MAX(glon(nx,j),elon)
    wlon=MIN(glon(1,j),wlon)
  END DO

!
!-----------------------------------------------------------------------
!
!  Test to see if maxlat and minlat are within physical limits....
!
!-----------------------------------------------------------------------
!

  IF (nlat > 90.0.OR.slat <= -90.0)THEN

    PRINT *,'maxlat is greater than 90. or minlat is less than ',       &
            '-90. degrees latitude'
    PRINT *,'maxlat,minlat',nlat,slat
    STOP

  END IF

!
!-----------------------------------------------------------------------
!
!  Print the bound of the lat/lons of the analysis grid...
!
!-----------------------------------------------------------------------
!

  PRINT *,'maxlat,minlat',nlat,slat
  PRINT *,'maxlon,minlon',elon,wlon

!
!-----------------------------------------------------------------------
!
!  Estimating the minumum data spacing in meters
!  using the two furthest northward data points.  These values
!  will be used to determine the largest number of data points
!  needed to determine the barnes analysis data buffer
!
!-----------------------------------------------------------------------
!

  CALL lltoxy(1,1,nlat,glon(1,ny),tema,temb)
  temb=glon(1,ny)+REAL(1./ndx)
  CALL lltoxy(1,1,nlat,temb,temc,temd)
  terdx=ABS(tema-temc)/sclfct

!
!-----------------------------------------------------------------------
!
!  Calculating the maximum terdx at the southern most point in the domain
!
!-----------------------------------------------------------------------
!

  CALL lltoxy(1,1,slat,glon(1,1),tema,temb)
  temb=glon(1,1)+REAL(1./ndx)
  CALL lltoxy(1,1,slat,temb,temc,temd)
  mterdx=ABS(tema-temc)/sclfct

!
!-----------------------------------------------------------------------
!
!  Calculating terdy at the northern most point in the domain
!
!-----------------------------------------------------------------------
!

  CALL lltoxy(1,1,nlat,glon(1,ny),tema,temb)
  tema=nlat+REAL(1./ndy)
  CALL lltoxy(1,1,tema,glon(1,ny),temc,temd)
  terdy=ABS(temb-temd)/sclfct

!
!-----------------------------------------------------------------------
!
!  Pring the terrain data spacings...
!
!-----------------------------------------------------------------------
!

  PRINT *,'terdx,terdy,mterdx',terdx,terdy,mterdx
  PRINT *,'2*mterdx=',2*mterdx,'2*terdy=',2*terdy

!
!-----------------------------------------------------------------------
!
!    Setting the Barnes influence factor and response to the desired
!    wdn wavelength set by the user in arpstern.input.
!
!-----------------------------------------------------------------------
!

  temb=MAX(dx,dy)
  tema=wdn*temb*sclfct

  IF (knot /= 0.0)THEN        ! temak is knot read from
                              ! arpstern.input
    temak=knot
  ELSE
    temak=-(tema/pi)*(tema/pi)*LOG(rdnot)
  END IF

!
!-----------------------------------------------------------------------
!
!  Calculate and plot the barnes response function
!
!-----------------------------------------------------------------------
!

  temb=tema/wdn
  temc=temb/sclfct
  CALL response(tema,temb,temc,temak,pi)

!
!-----------------------------------------------------------------------
!
!  Calculate nbufx and nbufy...
!
!-----------------------------------------------------------------------
!

  tema=SQRT(-LOG(tol)*temak)
  nbufx=INT(tema/(terdx*sclfct)+1)
  nbufy=INT(tema/(terdy*sclfct)+1)

!
!-----------------------------------------------------------------------
!
!  Print buffer sizes, working knot and cutoff radius....
!
!-----------------------------------------------------------------------
!

  PRINT *, 'nbufx,nbufy',nbufx,nbufy
  PRINT *, 'knot,temak,rc',knot,temak,tema

!-----------------------------------------------------------------------
!
!  Set up the limits of the latitude and longitude values for the
!  data area.
!
!-----------------------------------------------------------------------

  temc=REAL(1./ndy)
  temd=REAL(1./ndx)

  slat=slat-REAL((ipass+1)*nbufy*temc)
  wlon=wlon-REAL((ipass+1)*nbufx*temd)
  tema=slat-INT(slat)
  temb=wlon-INT(wlon)
  tema=INT(tema/temc)
  temb=INT(temb/temd)
  slat=REAL(INT(slat)+tema*temc)
  wlon=REAL(INT(wlon)+temb*temd)

  nlat=nlat+REAL((ipass+1)*nbufy*temc)
  elon=elon+REAL((ipass+1)*nbufx*temd)
  tema=nlat-INT(nlat)
  temb=elon-INT(elon)
!  tema=REAL(INT(tema/temc)+1.) !  TESTING ORIGINAL CODE
!  temb=REAL(INT(temb/temd)+1.) !  TESTING ORIGINAL CODE
  tema=INT(tema/temc)  !  TESTING NEW CODE
  temb=INT(temb/temd)  !  TESTING NEW CODE
  nlat=REAL(INT(nlat)+tema*temc)
  elon=REAL(INT(elon)+temb*temd)

!
!-----------------------------------------------------------------------
!
!  Print lat/lon bounds of the initial data field...
!
!-----------------------------------------------------------------------
!
  PRINT *,'Buffer adjusted nlat,slat,wlon,elon'
  PRINT *,nlat,slat,wlon,elon

!-----------------------------------------------------------------------
!
!  Determine ntx, nty the total number of data points needed
!  to perform the barnes analysis for the analysis domain.
!
!-----------------------------------------------------------------------

! original code
  nty=INT(REAL(nlat-slat)*ndy)+2    ! add one more point by Y. Wang.
  ntx=INT(REAL(elon-wlon)*ndx)+1
! testing new code...
!  nty=INT(REAL(nlat-slat)*ndy)     ! comment out by Y. Wang.
!  ntx=INT(REAL(elon-wlon)*ndx)

  PRINT *,'the number of terrain points in the x dir is ',ntx
  PRINT *,'the number of terrain points in the y dir is ',nty


  RETURN
END SUBROUTINE setgrid

!##################################################################
!##################################################################
!######                                                      ######
!######                SUBROUTINE COMPGRID                   ######
!######                                                      ######
!######                     Developed by                     ######
!######     Center for Analysis and Prediction of Storms     ######
!######                University of Oklahoma                ######
!######                                                      ######
!##################################################################
!##################################################################
!
!

SUBROUTINE compgrid(nx,ny,ndx,ndy,pi,                         &
           ntx,nty,nbufx,nbufy,slat,nlat,wlon,elon,temak,     &
           temc,temd,teme,temf,terdx,terdy,                   &
           xg1d,yg1d,xg,yg,glat,glon,closi,closj,             &
           xob,yob,doblat,doblon,closit,closjt)

!-----------------------------------------------------------------------
!
!  PURPOSE:
!
!  To compute the grid point locations for use by the chosen
!  data analysis technique.  Added due to f90 changes.
!
!
!-----------------------------------------------------------------------
!
!  AUTHOR: Dan Weber
!  11/28/2000
!
!-----------------------------------------------------------------------
!
!
!
!  INPUT :
!
!    nx       Number of grid points for the analysis grid e-w dir.
!    ny       Number of grid points for the analysis grid n-s dir.
!    ntx      number of e-w data points
!    nty      number of n-s data points
!    ndx      Number of data points in a 1 degree block (e-w)
!    ndy      Number of data points in a 1 degree block (n-s)
!    pi       The constant Pi
!
!-----------------------------------------------------------------------
!
!  COMMON BLOCK VARIABLES:
!
!-----------------------------------------------------------------------
!
!    analtype Analysis type (1=barnes,2=other,...)
!    mapproj  Type of map projection used to setup the analysis grid.
!             mapproj   = 1  Polar Stereographic projection
!                       = 2  Mercator projection
!                       = 3  Lambert
!    itertype Type of terrain desired (1=30 second, 2= 5 minute,
!                                      3= 1 degree)
!    rmsopt   Option for calc. the rms differences. (0=no,1=yes)
!    comtype  Computer type the arpstern11.f program will be run on
!             comtype  =  1 for IBM and other machine capable of
!                           declaring 2 byte integers.OR.rmsopt == 1
!                      =  4 for CRAY machines and other machines
!                           in which the minimum integer declaration
!                           is integer *8
!    knot     Shape parameter for the barnes weight function.
!    gamma    Shape factor for multi pass scheme
!    ipass    Total number of passes thru the barnes scheme
!    trulat   Latitude(s) at which the map projection is true
!             (degree N)
!    trulon   Longitude at which the map projection is true
!             (degree E)
!    sclfct   Map scale factor (m)
!
!    dx       Analysis grid spacing in the x-direction east-west (m)
!    dy       Analysis grid spacing in the y-direction north-south(m)
!    ctrlat   Latitude of the center of the analysis grid (deg. N)
!    ctrlon   Longitude of the center of the analysis grid (deg. E)
!    tol      Tolerance for the weight function used in determining
!             the cut-off radius or radius of influence
!    wdn      Wavelength in terms of grid points in which rdnot is
!             valid
!    rdnot    Initial Response for wdn length wave
!
!  LOCAL VARIABLES:
!
!    dincx    Analysis x-direction grid spacing normalized by the map
!             scale dincx=dx/sclfct
!    dincy    Analysis y-direction grid spacing normalized by the map
!             scale dincy=dy/sclfct
!    mterdx   Max. distance between e-w data pts. (m)
!    terdx    Distance in meters between data points in the e-w dir.
!             (m)
!    terdy    Distance in meters between data points in the n-s dir.
!             (m)
!
!  OUTPUT:
!
!    ntx      Number of data points needed in the east-west direction
!    nty      Number of data points needed in the north-south
!             direction
!    nbufx    Number of additional points needed to perform the
!             Barnes
!             analysis at the edges of the domain. e-w
!    nbufy    Number of additional points needed to perform the
!             Barnes
!             analysis at the edges of the domain. n-s
!    nlat     Latitude of the northern edge of the data area
!             (degrees N)
!    slat     Latitude of the southern edge of the data area
!             (degrees N)
!    elon     Longitude of the eastern edge of the data area
!             (degrees E)
!    wlon     Longitude of the western edge of the data area
!             (degrees E)
!    temak    Working value of knot (user or program specified)
!    xg1d     Analysis grid points e-w (1-d) in grid units
!             (=meters/sclfct)
!    yg1d     Analysis grid points n-s (1-d) in grid units
!             (=meters/sclfct)
!    xg       Analysis grid points in the e-w direction
!             (in grid units)
!    yg       Analysis grid points in the n-s direction
!             (in grid units)
!    glat     Latitude of the analysis data points
!    glon     Longitude of the analysis data points
!    closi    Array storing the data i index value corresponding to
!             the closest analysis grid point
!    closj    Array storing the data j index value corresponding to
!             the closest analysis grid point
!    xob      Analysis grid value in the east-west direction for the
!             data points (terrain data)
!    yob      Analysis grid value in the north-south direction for
!             the data points (terrain data)
!    doblat   Latitude of the data points (degrees north)
!    doblon   Longitude of the data points (degrees east)
!    closit   Array storing the terrain i index value corresponding
!             to the closest terrain point
!    closjt   Array storing the terrain j index value corresponding
!             to the closest terrain point
!
!
!-----------------------------------------------------------------------
!

!
!-----------------------------------------------------------------------
!
!  Variable Declarations.
!
!-----------------------------------------------------------------------

  IMPLICIT NONE
!
!-----------------------------------------------------------------------
!
!  INPUT:
!
!-----------------------------------------------------------------------
!

  INTEGER :: nx                ! Number of analysis grid points e-w
  INTEGER :: ny                ! Number of analysis grid points n-s
  INTEGER :: ndx               ! Number of data points (e-w)
  INTEGER :: ndy               ! Number of data points (n-s)
  REAL :: pi                   ! The constant Pi

!
!-----------------------------------------------------------------------
!
!  COMMON BLOCK VARIABLES:
!
!-----------------------------------------------------------------------
!

  INTEGER :: analtype          ! Analysis type
  INTEGER :: mapproj           ! Type of map projection used in
                               ! analysis
  INTEGER :: itertype          ! Desired terrain resolution for
                               ! analysis
  INTEGER :: rmsopt            ! Option for enabling the rms diff.
                               ! routine.
  INTEGER :: comtype           ! Computer type, =1 for ibm, =4 for
                               ! cray

  CHARACTER (LEN=80   ) :: tdatadir  ! Directory containing the terrain
                                     ! data
  INTEGER :: ltdir             ! Length of none-blank part of
                               ! tdatadir.

  REAL :: knot                 ! Influence parameter for the barnes
                               ! scheme
  REAL :: gamma                ! Barnes shape factor for multi pass
                               ! scheme
  INTEGER :: ipass             ! Total number of passes


  REAL :: trulat (2)           ! Latitude of true map projection
                               ! (deg N)
  REAL :: trulon               ! Longitude of true map projection
                               ! (deg E)
  REAL :: sclfct               ! Map scale factor (m)

  REAL :: dx                   ! Analysis grid spacing e-w direction
                               ! (m)
  REAL :: dy                   ! Analysis grid spacing n-s direction
                               ! (m)
  REAL :: ctrlat               ! Latitude of the center of the
                               ! analysis grid (deg. n)
  REAL :: ctrlon               ! Longitude of the center of the
                               ! analysis grid (deg. e)
  REAL :: tol                  ! Tolerance level for the weight
                               ! function
  REAL :: wdn                  ! Wavelength in terms of analysis grid
                               ! points for the rdnot response
  REAL :: rdnot                ! Initial Response for wdn length wave

!
!-----------------------------------------------------------------------
!
!  LOCAL VARIABLES:
!
!-----------------------------------------------------------------------
!

  REAL :: terdx                ! Distance in grid meters between e-w
                               ! data pts
  REAL :: terdy                ! Distance in grid meters between n-s
                               ! data pts

!
!-----------------------------------------------------------------------
!
!  OUTPUT:
!
!-----------------------------------------------------------------------
!

  INTEGER :: ntx               ! Actual number of data points needed
                               ! for data domain e-w
  INTEGER :: nty               ! Actual number of data points needed
                               ! for data domain n-s
  INTEGER :: nbufx             ! Number of buffer points in the x
                               ! direction
  INTEGER :: nbufy             ! Number of buffer points in the y
                               ! direction
  REAL :: nlat                 ! Max. Latitude of the data area
                               ! (degrees n)
  REAL :: slat                 ! Min. Latitude of the data area
                               ! (degrees n)
  REAL :: elon                 ! Max. Longitude of the data area
                               ! (degrees e)
  REAL :: wlon                 ! Min. Longitude of the data area
                               ! (degrees e)
  REAL :: temak                ! Working value of knot
  REAL :: xg1d(nx)             ! Analysis grid points e-w (1-d) in
                               ! grid units
  REAL :: yg1d(ny)             ! Analysis grid points n-s (1-d) in
                               ! grid units
  REAL :: xg(nx,ny)            ! 2-D analysis grid points east-west
  REAL :: yg(nx,ny)            ! 2-D analysis grid points north-south
  REAL :: glat(nx,ny)          ! Latitude of the analysis data points
  REAL :: glon(nx,ny)          ! Longitude of the analysis data
                               ! points
  INTEGER :: closi(nx,ny)      ! Array storing the data i index value
                               ! corresponding to the closest
                               ! analysis pt.
  INTEGER :: closj(nx,ny)      ! Array storing the data j index value
                               ! corresponding to the closest
                               ! analysis pt.
  REAL :: xob(ntx,nty)       ! Data set grid point values east-west
  REAL :: yob(ntx,nty)       ! Data set grid point values
                               ! north-south
  REAL :: doblat(ntx,nty)    ! Latitude of the data points
                               ! (degrees north)
  REAL :: doblon(ntx,nty)    ! Longitude of the data points
                               ! (degrees east)
  INTEGER :: closit(ntx,nty) ! array storing the data i index value
                               ! corresponding to the closest data
                               ! point
  INTEGER :: closjt(ntx,nty) ! array storing the data j index value
                               ! corresponding to the closest data
                               ! point

!
!-----------------------------------------------------------------------
!
!  Misc. local variables:
!
!-----------------------------------------------------------------------
!
  INTEGER :: i,j,k,l,m,n,itema,itemb
  REAL :: tema,temb,temc,temd,teme,temf
!
!-----------------------------------------------------------------------
!
!  Defining COMMON blocks:
!
!-----------------------------------------------------------------------

  COMMON /terrainc/ analtype,mapproj,itertype,rmsopt,comtype            &
                    ,tdatadir,ltdir
  COMMON /projectc/ trulat,trulon,sclfct
  COMMON /gridc/ dx,dy,ctrlat,ctrlon,tol,wdn,rdnot
  COMMON /barnesc/ knot,gamma,ipass

!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
!
!  Beginning of executable code...
!
!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

!
!-----------------------------------------------------------------------
!
!  Print lat/lon bounds of the initial data field...
!
!-----------------------------------------------------------------------
!

  temc=REAL(1./ndy)
  temd=REAL(1./ndx)

  PRINT *,'in compgrid nlat,slat,wlon,elon'
  PRINT *,nlat,slat,wlon,elon

  DO j=1,nty
    DO i=1,ntx
      doblat(i,j)=slat+REAL((j-1.)*temc)
      doblon(i,j)=wlon+REAL((i-1.)*temd)
    END DO
  END DO

!
!-----------------------------------------------------------------------
!
!  Print a sample of the initial data field lat/lon values...
!
!-----------------------------------------------------------------------
!

  PRINT *,'doblon'
  WRITE (*,'(8f9.3)') ((doblon(i,j),i=1,8),j=1,3)
  PRINT *,'doblat'
  WRITE (*,'(8f9.3)') ((doblat(i,j),i=1,8),j=1,3)

!-----------------------------------------------------------------------
!
!  Calculate the grid point values for the data
!
!-----------------------------------------------------------------------

  CALL lltoxy(ntx,nty,doblat,doblon,xob,yob)
!
!-----------------------------------------------------------------------
!
!  Print a sample of the initial data field grid values...
!
!-----------------------------------------------------------------------
!

  PRINT *,'xob'
  WRITE (*,'(6f12.1)') ((xob(i,j),i=1,7),j=1,3)
  PRINT *,'yob'
  WRITE (*,'(6f12.1)') ((yob(i,j),i=1,7),j=1,3)

!
!-----------------------------------------------------------------------
!
!  Setting the corners of the analysis grid in degrees lat/lon for the
!  purpose of plotting later.( glat and glon will be overwritten in
!  subroutine barnes)
!
!-----------------------------------------------------------------------
!

  temc=glat(1,ny)
  temd=glon(1,ny)
  teme=glat(nx,1)
  temf=glon(nx,1)

!
!-----------------------------------------------------------------------
!
!  Looking for the data point which is closest to the analysis
!  point (1,1).  These indicies will be used to find the remaining points
!
!-----------------------------------------------------------------------
!

  tema=10000000.
  DO l=1,nty/2
    DO k=1,ntx/2
      temb=SQRT((xg1d(1)-xob(k,l))*(xg1d(1)-xob(k,l))+                  &
                (yg1d(1)-yob(k,l))*(yg1d(1)-yob(k,l)))
      IF(temb < tema)THEN
        m=k
        n=l
        tema=temb
      END IF
    END DO
  END DO

!
!-----------------------------------------------------------------------
!
!  Calculate an increment which insures each closest point will be found
!
!-----------------------------------------------------------------------
!

  itema=INT(dx/terdx+3)
  itemb=INT(dy/terdy+3)

  DO j=1,ny
    DO i=1,nx
      tema=10000000.
      DO l=n-itemb,n+itemb
        DO k=m-itema,m+itema
          temb=SQRT((xg1d(i)-xob(k,l))*(xg1d(i)-xob(k,l))+              &
                    (yg1d(j)-yob(k,l))*(yg1d(j)-yob(k,l)))
          IF (temb < tema)THEN
            closi(i,j)=k
            closj(i,j)=l
            tema=temb
          END IF
        END DO
      END DO
      m=closi(i,j)
      n=closj(i,j)
    END DO
    m=closi(1,j)
    n=closj(1,j)
  END DO

!
!-----------------------------------------------------------------------
!
!  Print a sample of closi and closj...
!
!-----------------------------------------------------------------------
!

  PRINT *,'closi'
  WRITE (*,'(14i5)') ((closi(i,j),i=1,14),j=1,3)
  PRINT *,'closj'
  WRITE (*,'(14i5)') ((closj(i,j),i=1,14),j=1,3)

!
!-----------------------------------------------------------------------
!
!  Set the closit, closjt for the terrain data points analysis.
!
!-----------------------------------------------------------------------
!

  DO j=1,nty
    DO i=1,ntx
      closit(i,j)=i
      closjt(i,j)=j
    END DO
  END DO

!
!-----------------------------------------------------------------------
!
!  Expand the 1 dimensional grid arrays into 2-D arrays for use
!  in the barnes scheme.
!
!-----------------------------------------------------------------------
!

  DO j=1,ny
    DO i=1,nx
      xg(i,j)=xg1d(i)
      yg(i,j)=yg1d(j)
    END DO
  END DO


  RETURN
END SUBROUTINE compgrid


!##################################################################
!##################################################################
!######                                                      ######
!######                SUBROUTINE GETTER                     ######
!######                                                      ######
!######                     Developed by                     ######
!######     Center for Analysis and Prediction of Storms     ######
!######                University of Oklahoma                ######
!######                                                      ######
!##################################################################
!##################################################################
!

SUBROUTINE getter(ntx,nty,ndx,ndy,slat,nlat,wlon,elon,                &
           h,                                                           &
           nval,readdat)
!
!-----------------------------------------------------------------------
!
!  PURPOSE:
!
!  To retrieve elevation data from terrain data files and fill (if
!  needed) areas which are not covered by the user specified data set
!  .  This program will read the desired terrain resolution header
!  file, if the data is found the program will read the data in and
!  load the terrain array.  If the initial read does not find the
!  data avaliable, the program will read the header file for the next
!  best avaliable resolution set. This process will continue until
!  data has been found.  If no data exists for this particular 1
!  degree block (this can happen!!!) the program will stop.
!
!  Three terrain resolutions are currently supported by this code:
!
!       1.  30 second resolution data for the continental United
!           States(approx. 1km resolution)
!       2.  5 minute resolution data for most of North America and
!           some of Europe.
!           (approx. 10km resolution)
!       3.  1 degree resolution (30 minute in some locations) for the
!           entire globe.  (minus a few locations including the south
!           pole) (appros. 110km resolution)
!
!  The structure of this program is as follows: (eg. 30 second data
!  chosen)
!       1. Read 30 second header for a block match.  If block is
!          found, read the data file and continue to the next block.
!          If no 30 second data exists for this block, the 5 minute
!          header file will be read to find a block match.  If there
!          is a match, the data will be read, and expanded to fit the
!          size of the 30 second data array.  If there is no match
!          with the 5 minute header file, the 1 degree header file
!          will be read.  If there is a match the 1 degree data will
!          be read and expanded to fit the 30 second data structure.
!          If no data exists for the block in the 1 degree header
!          file the program will print a stop message and end.
!
!
!-----------------------------------------------------------------------
!
!  AUTHOR: Dan Weber
!  1/12/94
!
!  MODIFICATION HISTORY:
!
!  2/16/98, Dan Weber.
!  Bug fix for tema and temb for southern hemisphere indexing.
!
!  2000/02/03 Gene Bassett.
!  Moved Great Lakes terrain corrections into a subroutine.
!
!  11/29/2000 Dan Weber
!  Bug fix to the loading of the first terrain value along a longitude
!  circle.
!  Bug fix for data sets that span the 0 longitude line.
!
!-----------------------------------------------------------------------
!
!
!  INPUT :
!
!    ntx     Limit of terrain data points for use in the east-west dir.
!    nty     Limit of terrain data points for use in the north-south dir.
!    ndx      Number of terrain data points in the (east-west) direction
!             for a 1 degree block of data
!    ndy      Number of terrain data points in the (north-south) direction
!             for a 1 degree block of data
!    nlat     Northern extent of the buffered data set (degrees N)
!    slat     Southern extent of the buffered data set  (degrees N)
!    elon     Eastern extent of the buffered data set (degrees E)
!    wlon     Western extent of the buffered data set (degrees E)
!
!
!  COMMON BLOCK VARIABLES:
!
!    analtype Type of analysis variable requested
!    mapproj  Type of projection desired
!    itertype Type of terrain data requested
!    rmsopt   Rms error option
!    comtype  Type of computer this analysis is run on
!
!    tdatadir Directory containing the terrain data
!    ltdir    Length of none-blank part of tdatadir
!
!  LOCAL VARIABLES :
!
!    id       Array for reading header information from the data files
!    idim     East-west dimension of the readdat array in the readter sub.
!    jdim     North-south dimension of the readdat array in the readter sub.
!    n        Record number for use in readter
!    nfile2   Character string storing the data file name
!    rlength  Record length of the data set
!
!  OUTPUT:
!
!    h        Terrain data for use in the Barnes analysis scheme (m)
!
!
!  TEMPORARY ARRAYS:
!
!    nval     Temporary array which contains the terrain data (m)
!    readdat  Temporary array used to read the data file
!
!-----------------------------------------------------------------------
!
!  Variable Declarations.
!
!-----------------------------------------------------------------------

  IMPLICIT NONE

!
!-----------------------------------------------------------------------
!
!  INPUT:
!
!-----------------------------------------------------------------------
!

  INTEGER :: ntx              ! Limit of terrain data points e-w dir.
  INTEGER :: nty              ! Limit of terrain data points n-s dir.
  INTEGER :: ndx               ! Number of terrain data points in the e-w
  INTEGER :: ndy               ! Number of terrain data points in the n-s
  REAL :: nlat                 ! Northern extent of the buffered data set (N)
  REAL :: slat                 ! Southern extent of the buffered data set (N)
  REAL :: elon                 ! Eastern extent of the buffered data set (E)
  REAL :: wlon                 ! Western extent of the buffered data set (E)

!
!-----------------------------------------------------------------------
!
!  COMMON BLOCK VARIABLES:
!
!-----------------------------------------------------------------------
!

  INTEGER :: analtype          ! Type of analysis variable requested
  INTEGER :: mapproj           ! Type of projection desired
  INTEGER :: itertype          ! =1,2,or 3 for 30 sec,5 min, or 1 deg data
  INTEGER :: rmsopt            ! Rms error option
  INTEGER :: comtype           ! Computer type

  CHARACTER (LEN=80   ) :: tdatadir  ! Directory containing the terrain data
  INTEGER :: ltdir             ! Length of none-blank part of tdatadir.
!
!-----------------------------------------------------------------------
!
!  LOCAL VARIABLES:
!
!-----------------------------------------------------------------------
!

  INTEGER :: id(2)             ! Data file header array
  INTEGER :: idim              ! East-west dimension of the readdat array
                               ! in the readter subroutine
  INTEGER :: jdim              ! North-south dimension of the readdat array
                               ! in the readter subroutine
  INTEGER :: n                 ! Record number for use in readter
  CHARACTER (LEN=80) :: nfile2       ! Character storing the data file name
  INTEGER :: rlength           ! Length of the record in a data set

!
!-----------------------------------------------------------------------
!
!  OUTPUT:
!
!-----------------------------------------------------------------------
!

  REAL :: h(ntx,nty)         ! Terrain data for use in the analysis scheme,
                               ! includes the buffered area. (m)

!
!-----------------------------------------------------------------------
!
!  TEMPORARY ARRAYS:
!
!-----------------------------------------------------------------------
!

  INTEGER*2 nval(ndx,ndy)   ! Temporary array which contains the data (m)
  INTEGER*2 readdat(ndx,ndy)! Temporary array which used to read the file

!-----------------------------------------------------------------------
!
!  Misc. local variables:
!
!-----------------------------------------------------------------------
!
  REAL :: newslat,newnlat,newelon,newwlon,teme,temf
  INTEGER :: i,j,k,l,ll,inew,acount
  INTEGER :: tema,temb,temc,temd
  INTEGER :: ii,jj,kk,kbgn,kend,lbgn,lend,istat
!
!-----------------------------------------------------------------------
!
!-----------------------------------------------------------------------
!
!  COMMON block definition:
!
!-----------------------------------------------------------------------
!
  COMMON /terrainc/ analtype,mapproj,itertype,rmsopt,comtype            &
                    ,tdatadir,ltdir
!
!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
!
!  Beginning of executable code...
!
!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

  PRINT *,'entering getter'

!-----------------------------------------------------------------------
!
!  Setting the internal dimensions of the readdat array specific to the
!  type of data read.
!
!  NOTE: If the desired data resolution is 5 minutes, the 30 second data
!  file will not be accessed.  If the desired data resolution is 1 degree
!  the 5 minute and 30 second data sets will not be accessed.
!
!-----------------------------------------------------------------------

  PRINT *,'itertype=',itertype

  IF(itertype > 3.OR.itertype < 1)THEN
    PRINT *,'itertype is not supported, program will stop'
    STOP
  END IF

  OPEN(12,FILE=tdatadir(1:ltdir)//'dir30sec.hdr',                       &
          FORM='formatted',STATUS='old',IOSTAT=istat)
  IF( istat /= 0) THEN
    WRITE(6,'(/a,/a/)')                                                 &
        ' Error occured when opening terrain header file ',             &
        tdatadir(1:ltdir)//'dir30sec.hdr',' Program stopped in GETTER.'
    STOP
  END IF

  OPEN(13,FILE=tdatadir(1:ltdir)//'dir5min.hdr',                        &
          FORM='formatted',STATUS='old',IOSTAT=istat)
  IF( istat /= 0) THEN
    WRITE(6,'(/a,/a/)')                                                 &
        ' Error occured when opening terrain header file.',             &
        tdatadir(1:ltdir)//'dir5min.hdr',' Program stopped in GETTER.'
    STOP
  END IF


  OPEN(14,FILE=tdatadir(1:ltdir)//'dir1deg.hdr',                        &
          FORM='formatted',STATUS='old',IOSTAT=istat)
  IF( istat /= 0) THEN
    WRITE(6,'(/a,/a/)')                                                 &
        ' Error occured when opening terrain header file.',             &
        tdatadir(1:ltdir)//'dir1deg.hdr',' Program stopped in GETTER.'
    STOP
  END IF

!
!    new code added to fix a southern hemisphere indexing bug...
!    (if statements)

  IF(slat >= 0.0)THEN   ! compute the index for the southern lat.
    tema=INT((slat-INT(slat))*ndy)+1
  ELSE     !   compute the index for the southern lat.
    IF( ABS(slat-INT(slat)) < 0.000001)THEN
      tema = 1
    ELSE
      tema=ABS(INT((1.+slat-INT(slat))*ndy))+1
    END IF
  END IF
  IF(nlat >= 0.0)THEN   ! compute the index for the northern lat.
    temb=INT((nlat-INT(nlat))*ndy)+1
  ELSE     !   compute the index for the northern lat.
    IF(abs(nlat-INT(nlat)) < 0.000001) THEN
      temb = 1
    ELSE
      temb=ABS(INT((1.+nlat-INT(nlat))*ndy))+1
    END IF
  END IF

!  added/modified the following to fix a bug DBW, 11/29/2000  
  IF(wlon < 0.0)THEN
    IF(abs(wlon-INT(wlon)) < 0.000001) THEN
      temc = 1
    ELSE
      temc=ABS(INT((1.+wlon-INT(wlon))*ndx))+1
    END IF
  ELSE 
    temc=INT((wlon-INT(wlon))*ndx)+1
  END IF 

  IF(elon < 0.0)THEN
    IF(abs(elon-INT(elon)) < 0.000001) THEN
      temd = 1
    ELSE
      temd=ABS(INT((1.+elon-INT(elon))*ndx))+1
    END IF
  ELSE 
    temd=INT((elon-INT(elon))*ndx)+1
  END IF 

  PRINT *,'tema,temb,temc,temd',tema,temb,temc,temd

!
!-----------------------------------------------------------------------
!
!  Starting the main search and read loop.....
!
!-----------------------------------------------------------------------
!

  IF(slat < 0.0)THEN   ! testing for negative latitude limits...
    newslat=slat
    teme = slat-INT(slat) 
    IF(ABS(teme) > 0.000001 )newslat=slat-1.0
  ELSE
    newslat=slat
  END IF

  IF(nlat < 0.0)THEN   ! testing for negative latitude limits...
    newnlat=nlat
    teme = nlat-INT(nlat) 
    IF(ABS(teme) > 0.000001 )newnlat=nlat-1.0
  ELSE
    newnlat=nlat
  END IF

  IF(wlon < 0.0)THEN   ! testing for negative longitude limits...
    newwlon=wlon
    teme = wlon-INT(wlon) 
    IF(ABS(teme) > 0.000001 )newwlon=wlon-1.0
  ELSE
    newwlon=wlon
  END IF

  IF(elon < 0.0)THEN   ! testing for negative longitude limits...
    newelon=elon
    teme = elon-INT(elon)
    IF(ABS(teme) > 0.000001 )newelon=elon-1.0
  ELSE
    newelon=elon
  END IF


  jj=0
  DO j=INT(newslat),INT(newnlat)
    jj=jj+1
    ii=0
    acount = 0
    DO i=INT(newwlon),INT(newelon)
      ii=ii+1

      IF(i >= 360.AND.acount == 0)THEN  ! need to rewind the header files...
        acount = 1
        IF(itertype == 3)THEN      ! rewind dir1deg.hdr only...
          REWIND(14)
        ELSE IF(itertype == 2)THEN ! rewind dir5min and dir1deg.hdrs
          REWIND(13)
          REWIND(14)
        ELSE IF(itertype == 1)THEN ! rewind all header files...
          REWIND(12)
          REWIND(13)
          REWIND(14)
        END IF
      END IF

!
!-----------------------------------------------------------------------
!
!  Testing for values of longitude greater than 359.
!
!-----------------------------------------------------------------------
!
      inew=i

      IF(i >= 360) inew=inew-360

!  start of bug fix area for data spanning the 0 longitude line...
!  DBW 11/29/2000

      IF(i < 0) inew=inew+360

      IF(int(wlon) < 0.AND.(i >= 0.AND.i-1 <  0))THEN
        IF(itertype.eq.3)then      ! rewind dir1deg.hdr only...
          rewind(14)
        ELSE IF(itertype ==  2)then ! rewind dir5min and dir1deg.hdrs
          rewind(13)
          rewind(14)
        ELSE IF(itertype == 1)then ! rewind all header files...
          rewind(12)
          rewind(13)
          rewind(14)
        ENDIF
      ENDIF
!  end of bug fix area  DBW 11/29/2000.

      PRINT *,'looking for latitude, longitude',j,inew

      IF(itertype == 1)THEN       ! reading the 30 second header file
        1        READ(12,'(3i6)',END=2) id(1),id(2),n
        IF(id(1) == j.AND.id(2) == inew)THEN    ! testing for a match
          nfile2=tdatadir(1:ltdir)//'dir30sec.dat '
          PRINT *,'we have a match reading ',nfile2
          idim=120
          jdim=120
          rlength=28800*comtype
          CALL readter(n,idim,jdim,ndx,ndy,nfile2,rlength,              &
                       readdat,nval)
          GO TO 99               ! Data has been read, add to the h array...
        ELSE                     ! no match yet read header file again...
          GO TO 1
        END IF
      END IF

      2      PRINT *,'block not found in dir30sec.hdr or data not desired'
      REWIND (12)

      IF(itertype == 2.OR.itertype == 1)THEN  ! testing for correct itertype
        3        READ(13,'(3i6)',END=4) id(1),id(2),n
        IF(id(1) == j.AND.id(2) == inew)THEN  ! testing for a match (5 minute)
          nfile2=tdatadir(1:ltdir)//'dir5min.dat '
          PRINT *,'reading ',nfile2,n
          idim=12
          jdim=12
          rlength=288*comtype
          CALL readter(n,idim,jdim,ndx,ndy,nfile2,rlength,              &
                       readdat,nval)
          GO TO 99
        ELSE
          GO TO 3
        END IF
      END IF

      4      PRINT *,'block not found in dir5min.hdr or data not desired'
      REWIND (13)

!
!-----------------------------------------------------------------------
!
!  Reading the dir1deg header file.
!
!-----------------------------------------------------------------------
!

      PRINT *,'reading dir1deg.hdr file'
      5      READ(14,'(3i6)',END=6) id(1),id(2),n
      IF(id(1) == j.AND.id(2) == inew)THEN      ! testing for a match...
        nfile2=tdatadir(1:ltdir)//'dir1deg.dat'
        PRINT *,'reading ',nfile2
        idim=2
        jdim=2
        rlength=8*comtype
        CALL readter(n,idim,jdim,ndx,ndy,nfile2,rlength,                &
                     readdat,nval)
        print *,nval(1,1),nval(2,1),nval(1,2),nval(2,2)
        print *,readdat(1,1),readdat(2,1),readdat(1,2),readdat(2,2)
        GO TO 99             ! update the terrain array.....
      ELSE
        GO TO 5              ! read the header file again......
      END IF

      6      PRINT *,'block not found in dir1deg.hdr,no data was found ', &
                     'for this block, the program will stop'
      STOP

!
!-----------------------------------------------------------------------
!
!  Update the terrain array (h), unpack, and convert to meters.
!
!-----------------------------------------------------------------------
!

      99     PRINT *,'adding nval to the h array'

!
!-----------------------------------------------------------------------
!
!  Setting up the ranges for adding each nval to the h array....
!
!-----------------------------------------------------------------------
!


      IF(jj == 1.AND.j == INT(newnlat))THEN
        lbgn=tema
        lend=temb
        ll=0
      ELSE IF(jj == 1)THEN
        lbgn=tema
        lend=ndy
        ll=0
      ELSE IF (j == INT(newnlat))THEN
        lbgn=1
        lend=temb
        ll=ndy-tema+1+(jj-2)*ndy
      ELSE
        lbgn=1
        lend=ndy
        ll=ndy-tema+1+(jj-2)*ndy
      END IF

      DO l=lbgn,lend
        ll=ll+1

        IF(ii == 1.AND.i == INT(newelon))THEN
          kbgn=temc
          kend=temd
          kk=0
        ELSE IF(ii == 1)THEN
          kbgn=temc
          kend=ndx
          kk=0
        ELSE IF(i == INT(newelon))THEN
          kbgn=1
          kend=temd
          kk=ndx-temc+1+(ii-2)*ndx
        ELSE
          kbgn=1
          kend=ndx
          kk=ndx-temc+1+(ii-2)*ndx
        END IF

        DO k=kbgn,kend
          kk=kk+1
          h(kk,ll)=REAL(nval(k,l)*20-4000)   ! units are in meters...
!         print *,'kk ll and h = ',kk,ll,h(kk,ll),nval(k,l),k,l
        END DO
      END DO
           print *,'kbgn,kend,lbgn,lend ',kbgn,kend,lbgn,lend
    END DO
  END DO

  CLOSE(12)
  CLOSE(13)
  CLOSE(14)

  RETURN
END SUBROUTINE getter



!##################################################################
!##################################################################
!######                                                      ######
!######                SUBROUTINE READTER                    ######
!######                                                      ######
!######                     Developed by                     ######
!######     Center for Analysis and Prediction of Storms     ######
!######                University of Oklahoma                ######
!######                                                      ######
!##################################################################
!##################################################################
!

SUBROUTINE readter(n,idim,jdim,ndx,ndy,nfile2,rlength,                  &
           readdat,nval)

!-----------------------------------------------------------------------
!
!  PURPOSE:
!
!  To retrieve terrain data from the selected data files
!  This program will read either the 30 second, 5 minute, or 1 degree
!  terrain data base files in a direct access format.
!  (original sources, DMA_ELEV.DAT or elev.dat data files obtain from NCAR)
!
!  NOTE: these original data files have been modified to run more
!  efficiently (put in direct access format by dir30sec.f, dir5min.f and
!  dir1deg.f).
!
!-----------------------------------------------------------------------
!

!
!-----------------------------------------------------------------------
!
!  Variable definitions
!
!-----------------------------------------------------------------------
!
!  INPUT:
!
!    n       value of the record number
!    idim    x dimension of the readdat array
!    jdim    y dimension of the readdat array
!    ndx     Number of terrain data points in the e-w direction
!            for the DESIRED data resolution
!    ndy     Number of terrain data points in the n-s direction
!            for the DESIRED data resolution
!    nfile2  Filename of the data set to be accessed
!    rlength Length of the record to be read from the data set
!    readdat Array which will read the data file
!
!  OUTPUT:
!
!    nval    Array in which the terrain values will be stored
!
!-----------------------------------------------------------------------
!
!  Variable declarations
!
!-----------------------------------------------------------------------

  IMPLICIT NONE

!
!-----------------------------------------------------------------------
!
!  INPUT:
!
!-----------------------------------------------------------------------
!

  INTEGER :: n                    ! Specific value of the record number
  INTEGER :: idim                 ! East-west readdat dimension
  INTEGER :: jdim                 ! North-south readdat dimension
  INTEGER :: ndx                  ! East-west nval dimension
  INTEGER :: ndy                  ! North-south nval dimension
  CHARACTER (LEN=*        ) :: nfile2 ! Data set filename
  INTEGER :: rlength              ! Length of the data record
  INTEGER*2 readdat(idim,jdim) ! Array which will read the data file

!
!-----------------------------------------------------------------------
!
!  OUTPUT:
!
!-----------------------------------------------------------------------
!

  INTEGER*2 nval(ndx,ndy)      ! Array of terrain data for the block

!
!-----------------------------------------------------------------------
!
!  Misc. local variables:
!
!-----------------------------------------------------------------------
!
  INTEGER :: itema,itemb
  REAL    :: tema,temb,temc,temd
  INTEGER :: k,l,istat
!
!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
!
!  Beginning of executable code...
!
!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

  OPEN(11,FILE=nfile2,ACCESS='direct',FORM='unformatted',               &
          STATUS='old',RECL=rlength,IOSTAT=istat)

  IF( istat /= 0) THEN
    WRITE(6,'(/a,/a,a/)')                                               &
        'Error occured when opening terrain data file.',                &
        nfile2,', Program stopped in READTER.'
    STOP
  END IF

  READ(11,REC=n,ERR=991) readdat

  CLOSE (11)

!
!-----------------------------------------------------------------------
!
!  Filling the nval array
!
!-----------------------------------------------------------------------
!

  IF(idim < ndx)THEN   ! if true, nval will be filled stepwise

    temc=REAL(idim/REAL(ndx))
    temd=REAL(jdim/REAL(ndy))
    DO l=1,ndy
      DO k=1,ndx
        tema=REAL((k-1)*temc)
        temb=REAL((l-1)*temd)
        itema=INT(tema)
        itemb=INT(temb)
        nval(k,l)=readdat(1+itema,1+itemb)
      END DO
    END DO

  ELSE                 ! direct fill...

    DO l=1,ndy
      DO k=1,ndx
        nval(k,l)=readdat(k,l)
      END DO
    END DO

  END IF

  RETURN

  991  PRINT*,'Error occurred when reading data ',nfile2,               &
              ' in READTER. Program stopped in READTER.'
  STOP

END SUBROUTINE readter


!##################################################################
!##################################################################
!######                                                      ######
!######                SUBROUTINE BARNES                     ######
!######                                                      ######
!######                     Developed by                     ######
!######     Center for Analysis and Prediction of Storms     ######
!######                University of Oklahoma                ######
!######                                                      ######
!##################################################################
!##################################################################
!

SUBROUTINE barnes(ipnum,ibgn,iend,jbgn,jend,nx,ny,ntx,nty,            &
           nbufx,nbufy,temak,zbt,za,                                  &
           zga,                                                       &
           xg,yg,xob,yob,closi,closj,tem1,tem2)

!
!-----------------------------------------------------------------------
!
!  PURPOSE:
!
!  Perform a Barnes analysis on a data base for a prespecified
!  knot and gamma (multipass scheme).  The user can specific knot, the
!  influence factor, in arpstern.input or allow the program to calculate
!  one given specified parameters (in setgrid).  The cutoff radius is
!
!         rc=(ln(rdnot) * KNOT)**0.5
!
!  The Barnes weight function is w = EXP(-r**2/knot)
!  where r is the distance between the analysis point and the data
!  point.
!
!-----------------------------------------------------------------------
!
!  AUTHOR: Dan Weber
!  1/13/94
!
!-----------------------------------------------------------------------

!
!  INPUT :
!

!    ipnum    Current pass number
!    ibgn     Starting index for the analysis (e-w)
!    iend     Ending index for the analysis (e-w)
!    jbgn     Starting index for the analysis (n-s)
!    jbgn     Ending index for the analysis (n-s)
!    nx       Number of analysis grid points in the east-west direction
!    ny       Number of analysis grid points in the north-south direction
!    ntx      Limit of number of data points e-w
!    nty      Limit of number of data points n-w
!    nbufx    Number of additional data points outside the model
!             domain used on the calculation of the boundaries. east-west
!    nbufy    Number of additional data points outside the model
!             domain used on the calculation of the boundaries. north-south
!    temak    Working value of knot (set in setgrid subroutine)
!
!    zbt      Analyzed data at analysis points obtained
!             from performing a barnes analysis at the data points.
!    za       Data for the buffered and analysis areas (m)
!
!    xg       Analysis grid points east-west direction (grid units)
!
!    yg       Analysis grid points north-south direction (grid units)
!
!    xob      Data set grid point values east-west direction
!    yob      Data set grid point values north-south direction
!
!    closi    Array storing the data i index value corresponding to
!             the closest analysis grid point
!    closj    Array storing the data j index value corresponding to
!             the closest analysis grid point
!
!  COMMON block variables:
!
!    knot     Shape parameter for the barnes weight function.
!    gamma    Response factor for multiple passes of the Barnes analysis
!    ipass    Total number of passes to be performed
!
!  OUTPUT:
!
!    zga      Analyzed data field. (m)
!
!
!  TEMPORARY ARRAYS
!
!    tem1     Temporary working array (barnes weight function)
!    tem2     Temporary working array
!
!
!-----------------------------------------------------------------------
!
!  Variable Declarations.
!
!-----------------------------------------------------------------------
!
  IMPLICIT NONE

!-----------------------------------------------------------------------
!
!  INPUT:
!
!-----------------------------------------------------------------------
!
  INTEGER :: ipnum             ! Current pass number
  INTEGER :: ibgn              ! starting index for the analysis (e-w)
  INTEGER :: iend              ! ending index for the analysis (e-w)
  INTEGER :: jbgn              ! starting index for the analysis (n-s)
  INTEGER :: jend              ! ending index for the analysis (n-s)
  INTEGER :: nx                ! Number of analysis grid points (e-w)
  INTEGER :: ny                ! Number of analysis grid points (n-s)
  INTEGER :: ntx              ! Limit of data points e-w
  INTEGER :: nty              ! Limit of data poitns n-s
  INTEGER :: nbufx             ! Number of buffer data points e-w
  INTEGER :: nbufy             ! Number of buffer data points n-s
  REAL :: temak                ! Working value of knot

  REAL :: zbt(ntx,nty)       ! Background data field (not required)
                               ! at the data points. (m)
  REAL :: za(ntx,nty)        ! Data base for use in the Barnes
                               ! analysis for the buffered area. (m)
  REAL :: xg(nx,ny)            ! Analysis grid points east-west
                               ! direction (analysis grid units)
  REAL :: yg(nx,ny)            ! Analysis grid points north-south
                               ! direction (analysis grid units)

  REAL :: xob(ntx,nty)       ! Data set grid point values east-west
                               ! direction
  REAL :: yob(ntx,nty)       ! Data set grid point values north-south
                               ! direction
  INTEGER :: closi(nx,ny)      ! array storing the data i index value
                               ! corresponding to the closest grid pt
  INTEGER :: closj(nx,ny)      ! array storing the data j index value
                               ! corresponding to the closest grid pt

!
!-----------------------------------------------------------------------
!
!  COMMON block variables:
!
!-----------------------------------------------------------------------
!

  REAL :: knot                 ! Shape parameter for the barnes scheme
  REAL :: gamma                ! Response factor for multiple pass Barnes
  INTEGER :: ipass             ! Total number of passes to be performed

!
!-----------------------------------------------------------------------
!
!  OUTPUT:
!
!-----------------------------------------------------------------------
!

  REAL :: zga(nx,ny)           ! Analyzed data base for use by the user (m)

!
!-----------------------------------------------------------------------
!
!  TEMPORARY ARRAYS
!
!-----------------------------------------------------------------------
!

  REAL :: tem1(nx,ny)          ! Temporary working array (Barnes Weights)
  REAL :: tem2(nx,ny)          ! Temporary working array

!
!-----------------------------------------------------------------------
!
!  Misc. local variables:
!
!-----------------------------------------------------------------------
!
  INTEGER :: i,j,m,n,in,jn
  REAL :: w,tema,temb,temc
!
!-----------------------------------------------------------------------
!
!  COMMON block definition:
!
!-----------------------------------------------------------------------
!
  COMMON /barnesc/ knot,gamma,ipass
!
!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
!
!  Beginning of executable code...
!
!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

  IF(ipnum == 1)THEN
    tema=1.0
    DO j=jbgn,jend
      DO i=ibgn,iend
        zga(i,j)=0.0
      END DO
    END DO
    DO j=1,nty
      DO i=1,ntx
        zbt(i,j)=0.0
      END DO
    END DO
  ELSE
    tema=gamma
  END IF

  DO j=jbgn,jend
    DO i=ibgn,iend
      tem1(i,j)=0.0
      tem2(i,j)=0.0
    END DO
  END DO

!
!-----------------------------------------------------------------------
!
!  BEGIN THE BARNES ANALYSIS***********
!
!-----------------------------------------------------------------------
!
!
!-----------------------------------------------------------------------
!
!  Print the bounds of the barnes analysis and current pass number.
!
!-----------------------------------------------------------------------
!

  PRINT *,'ibgn,iend,jbgn,jend',ibgn,iend,jbgn,jend
  PRINT *,'current barnes pass ',ipnum
  PRINT *,'closi(1,ny),closj(1,ny),nbufx,nbufy',  &
           closi(1,ny),closj(1,ny),nbufx,nbufy

  tema=temak*tema
  DO n=-nbufy,nbufy
    DO m=-nbufx,nbufx
      DO j=jbgn,jend
        DO i=ibgn,iend
          in=closi(i,j)+m
          jn=closj(i,j)+n
          temb=xg(i,j)-xob(in,jn)
          temc=yg(i,j)-yob(in,jn)
          w=EXP(-(temb*temb+temc*temc)/tema)
          tem2(i,j)=(za(in,jn)-zbt(in,jn))*w+tem2(i,j)
          tem1(i,j)=tem1(i,j)+w
        END DO
      END DO
    END DO
  END DO

!
!-----------------------------------------------------------------------
!
!  Summing up the weights from each observation inside the rc
!  and dividing by the total weight for each analysis grid point.
!
!-----------------------------------------------------------------------
!

  DO j=jbgn,jend
    DO i=ibgn,iend
      zga(i,j)=zga(i,j)+tem2(i,j)/tem1(i,j)
    END DO
  END DO

  PRINT *,'leaving the barnes subroutine'

  RETURN
END SUBROUTINE barnes

!##################################################################
!##################################################################
!######                                                      ######
!######                SUBROUTINE RMSDIF                     ######
!######                                                      ######
!######                     Developed by                     ######
!######     Center for Analysis and Prediction of Storms     ######
!######                University of Oklahoma                ######
!######                                                      ######
!##################################################################
!##################################################################
!

SUBROUTINE rmsdif(ibgn,iend,jbgn,jend,ntx,nty,tema,temb,temc,         &
                  temd,glab,f1,f2,diff)
!
!-----------------------------------------------------------------------
!
!  PURPOSE:
!
!  Calculate and plot the root mean square difference between
!  two data fields.
!
!-----------------------------------------------------------------------
!
!  AUTHOR:  Dan Weber
!  1/12/94
!
!-----------------------------------------------------------------------
!
!  INPUT:
!
!    ibgn     Starting i index for the differencing loop
!    iend     Ending i index for the differencing loop
!    jbgn     Starting j index for the differencing loop
!    jend     Ending j index for the differencing loop
!
!    ntx      Dimension in the e-w direction for the f1,f2, diff arrays
!    nty      Dimension in the n-s direction for the f1,f2, diff arrays
!
!    tema     Latitude of the northwest corner of the area
!    temb     Longitude of the northwest corner of the area
!    temc     Latitude of the southwest corner of the area
!    temd     Longitude of the southwest corner of the area
!    glab     Label for the difference field plot
!
!    f1       First data array
!    f2       Second data array
!
!
!  OUTPUT:
!
!    diff     Root mean square difference field between f1 and f2
!
!-----------------------------------------------------------------------
!
!  Variable Declarations.
!
!-----------------------------------------------------------------------

  IMPLICIT NONE

  INTEGER :: ibgn              ! Starting i index for the differencing loop
  INTEGER :: iend              ! Ending i index for the differencing loop
  INTEGER :: jbgn              ! Starting j index for the differencing loop
  INTEGER :: jend              ! Ending j index for the differencing loop

  INTEGER :: ntx              ! Dimension in the e-w direction for f1 and f2
  INTEGER :: nty              ! Dimension in the n-s direction for f1 and f2

  REAL :: tema                 ! Lat of the northwest corner of the area
  REAL :: temb                 ! Lon of the northwest corner of the area
  REAL :: temc                 ! Lat of the southwest corner of the area
  REAL :: temd                 ! Lon of the southwest corner of the area
  CHARACTER (LEN=65) :: glab         ! Label for the difference field plot

  REAL :: f1(ntx,nty)        ! Array containing field number 1
  REAL :: f2(ntx,nty)        ! Array containing field number 2


  REAL :: diff(ntx,nty)      ! Root mean square difference field
                               ! diff=sqrt((f1-f2)*(f1-f2))
!
!-----------------------------------------------------------------------
!
!  Misc. local variables:
!
!-----------------------------------------------------------------------
!
  INTEGER :: i,j
  REAL :: cint
!
!-----------------------------------------------------------------------
!
!  Defining COMMON blocks:
!
!-----------------------------------------------------------------------
!
  REAL :: trulat (2)           ! Latitude of true map projection
                               ! (deg N)
  REAL :: trulon               ! Longitude of true map projection
                               ! (deg E)
  REAL :: sclfct               ! Map scale factor (m)

  COMMON /projectc/ trulat,trulon,sclfct
!
!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
!
!  Beginning of executable code...
!
!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
!
  PRINT *,'ibgn,iend,jbgn,jend in rmsdiff'
  PRINT *,ibgn,iend,jbgn,jend
  DO j=jbgn,jend
    DO i=ibgn,iend
      diff(i,j)=SQRT((f1(i,j)-f2(i,j))*(f1(i,j)-f2(i,j)))
    END DO
  END DO

  cint=-4.
  CALL plot((iend-ibgn)+1,(jend-jbgn)+1,ntx,8,trulat,trulon,cint,      &
            tema,temb,temc,temd,glab,diff(ibgn,jbgn))

  RETURN
END SUBROUTINE rmsdif


!##################################################################
!##################################################################
!######                                                      ######
!######                SUBROUTINE RESPONSE                   ######
!######                                                      ######
!######                     Developed by                     ######
!######     Center for Analysis and Prediction of Storms     ######
!######                University of Oklahoma                ######
!######                                                      ######
!##################################################################
!##################################################################
!

SUBROUTINE response(tema,temb,temc,temak,pi)
!
!-----------------------------------------------------------------------
!
!  PURPOSE:
!
!  Calculate the final response function/curve for the Barnes analysis
!  Details of this method of calculating the response function
!  can be found in Koch et.al. JCAM Sept. 1983.
!
!-----------------------------------------------------------------------
!
!  AUTHOR: Dan Weber
!  12/6/93
!
!-----------------------------------------------------------------------
!
!  INPUT :
!
!
!    tema     Wdn*max grid spacing(dx or dy)*sclfct
!    temb     max grid spacing*sclfct
!    temc     max grid spacing
!    temak    Working value of knot (set in setgrid)
!    pi       The value pi
!
!
!  COMMON BLOCK VARIABLES:
!
!    knot     Barnes response shape factor
!    gamma    Multipass shape adjustment
!    ipass    Total passes for the Barnes scheme
!
!
!  OUTPUT:
!
!    d1star   response function for the barnes analysis
!
!
!-----------------------------------------------------------------------
!
!  Variable Declarations.
!
!-----------------------------------------------------------------------

  IMPLICIT NONE
!
!-----------------------------------------------------------------------
!
!  INPUT:
!
!-----------------------------------------------------------------------
!
  REAL :: tema                 ! Wdn*max spacing*sclfct
  REAL :: temb                 ! Max spacing*sclfct
  REAL :: temc                 ! max data or grid spacing
  REAL :: temak                ! Working value of knot (set in setgrid)
  REAL :: pi                   ! The value pi

  INTEGER :: inum              ! Number of points used to define the response
                               ! curve.

  PARAMETER (inum=35)
  REAL :: dnot(inum)           ! response function for the first pass of
                               ! the barnes analysis
  REAL :: xplot(inum)          ! x axis for response curve plot

!
!-----------------------------------------------------------------------
!
!  COMMON BLOCK VARIABLES:
!
!-----------------------------------------------------------------------
!

  REAL :: knot                 ! Barnes response shape factor
  REAL :: gamma                ! Multipass shape adjustment
  INTEGER :: ipass             ! Total passes for the Barnes scheme


!
!-----------------------------------------------------------------------
!
!  OUTPUT:
!
!-----------------------------------------------------------------------
!

  REAL :: dnstar(inum)         ! response function for the barnes analysis

!
!-----------------------------------------------------------------------
!
!  TEMPROARY ARRAYS:
!
!-----------------------------------------------------------------------
!
  REAL :: tem1(inum)
!
!-----------------------------------------------------------------------
!
!  Misc. local variables:
!
!-----------------------------------------------------------------------
!
  INTEGER :: i,n
  REAL :: alamnot,akstar,alam,alamstr,tem
  CHARACTER (LEN=75) :: glab
!
!-----------------------------------------------------------------------
!
!  COMMON BLOCK DEFINITION:
!
!-----------------------------------------------------------------------

  COMMON /barnesc/ knot,gamma,ipass

!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
!
!  Beginning of executable code...
!
!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

  alamnot=tema
  akstar=temak/(alamnot*alamnot)
  PRINT *,'tema,temak in response subroutine'
  PRINT *,tema,temak

!
!-----------------------------------------------------------------------
!
!  Calculate the initial response....
!
!-----------------------------------------------------------------------
!

  DO n=1,inum
    alam=REAL(n)*temb
    alamstr=alam/alamnot
    tem= -akstar*(pi/alamstr)*(pi/alamstr)
    dnot(n)=EXP(tem)
    xplot(n)=n
  END DO


  IF(ipass == 1)THEN      ! set dnot = dnstar and plot

    DO n=1,inum
      dnstar(n)=dnot(n)
    END DO

  ELSE IF(ipass == 2)THEN  ! calculate D1* = dnstar

    DO n=1,inum
      tem=dnot(n)**gamma
      dnstar(n)=dnot(n)+tem-tem*dnot(n)
    END DO

  ELSE                    ! calculate Dn* = dnstar

    DO n=1,inum
      dnstar(n)=0.0
      tem1(n)=1.0
    END DO

    DO i=2,ipass-1

      tem=gamma*(i-1)
      DO n=1,inum
        tem1(n)=(1.0-dnot(n)**tem)*tem1(n)
      END DO

      tem=i*gamma
      DO n=1,inum
        dnstar(n)=(dnot(n)**tem)*tem1(n)+dnstar(n)
      END DO
    END DO

    DO n=1,inum
      dnstar(n)=dnot(n)+(1.0-dnot(n))*((dnot(n)**gamma)+dnstar(n))
    END DO

  END IF

!-----------------------------------------------------------------------
!
!  Print the response for the first and last passes.
!
!-----------------------------------------------------------------------

  PRINT *,'wavelength (dx) first pass(dnot) last pass(dnotstar)'
  DO n=1,inum
    PRINT *,n,dnot(n),dnstar(n)
  END DO

!-----------------------------------------------------------------------
!
!  Set up the labels for the plot axis
!
!-----------------------------------------------------------------------

  CALL agsetc('LABEL/NAME.','T')
  CALL agseti('LINE/NUMBER.',100)
  CALL agseti('LINE/MAXIMUM.',70)
  CALL agsetc('LABEL/NAME.','B')
  CALL agseti('LINE/NUMBER.',-100)
  CALL agsetc('LINE/TEXT.','Wavelength (x*max grid spacing)')
  CALL agsetc('LABEL/NAME.','L')
  CALL agseti('LINE/NUMBER.',100)
  CALL agsetc('LINE/TEXT.','Response (g/f)')
  WRITE(glab,70) ipass,gamma,temc
  70   FORMAT('Barnes Response Function',' Pass # =',i2,                &
        ' Gamma =',f5.2,' max. spacing=',f8.1)
  CALL ezxy(xplot,dnstar,inum,glab)
  RETURN
END SUBROUTINE response
