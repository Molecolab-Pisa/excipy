! excipy: machine learning models for a fast estimation of excitonic Hamiltonians
! Copyright (C) 2022 excipy authors
! 
! This program is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
! 
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
! 
! You should have received a copy of the GNU General Public License
! along with this program.  If not, see <https://www.gnu.org/licenses/>.
subroutine TMu(mu,alpha,thole,NNlist,CRD,NAt,NNmax,IScreen,E)
!
!  Computes E = T*mu, where T is the MMPol matrix (excluding the diagonal)
!  and mu are input dipoles. 
!
!  Some elements of T are zeroed or screened according to the 
!  Thole smeared dipole formulation.
!
!  Distance-dependent screening factors and factors for nearest neighbours 
!  are taken from the AL model of Wang et al. JPC B (2011), 115, 3091
!
implicit None
!
!NAt=num of atoms , NNmax=maximum number of neighbors
integer,                         intent(in)  :: NAt,NNmax
! IScreen=screening function form (linear form, exponential form)
integer,                         intent(in)  :: IScreen 
! mu=input dipoles, CRD=coordinates
real*8,  dimension(3,NAt),       intent(in)  :: mu,CRD    
!alpha=polarizability
real*8,  dimension(NAt),         intent(in)  :: alpha 
real*8,  dimension(NAt),         intent(in)  :: thole
! real*8, allocatable :: thole(:)
!NNlist=list of the nearest-neighbors (1-2, 1-3)
integer, dimension(NNMax,NAt),   intent(in)  :: NNlist 
! E = output field
real*8,  dimension(3,NAt),       intent(out) :: E
!
integer :: I,J,L,M,istat
logical :: DoInter !DoInter=false if the atoms are nearest-neighbors
!
real*8  :: R, R2, R3, R5, s, v, v3, lambda3, lambda5, fExp, ef, rm3, rm5, scd
real*8, dimension(3) :: Rvec, term3, term5, efi !efi=electric field at atom i
! 
real*8, parameter :: zero=0.0d0, one=1.0d0, three=3.0d0, four=4.0d0, sixth=1.0d0/6.0d0 
real*8, parameter :: wangal=2.5874d0

! Compute vector screening
! allocate (thole(nat),stat=istat)
! if (istat.ne.0) then
!   write(6,*) 'error in memory allocation in TMu'
!   stop
! end if
!!
! fA = sqrt(wangal)
! do i = 1, nat
!   thole(i) = fA*alpha(i)**sixth
! end do
!!
E = zero

! Set the parallel environment using OpenMP
!$omp parallel do default(shared) private(I,J,RVec,R2,R,R3,R5,s,v,v3,lambda3,lambda5,rm3,rm5, &
!$omp                                     ef,fExp,scd,term3,term5) reduction(+:efi)
Do I = 1,NAt
    efi = zero
    Do J = 1,NAt
      ! Decide whether to do the calc
      ! If I==J or I and J are nearest neighbors skip calc
      If ( .not.DoInter(I,J,NNlist,NAt,NNMax) ) then 
        Cycle
      EndIf

      ! Compute vector distance r_I - r_J , r_IJ, r_IJ^3, etc
      Rvec(:) = CRD(:,I) - CRD(:,J)

      R2 = Rvec(1)*Rvec(1) + Rvec(2)*Rvec(2) + Rvec(3)*Rvec(3)
      R  = sqrt(R2)

      R3 = R*R2
      R5 = R3*R2

      ! Compute screening 
!     s = wangal*( alpha(I)*alpha(J) ) ** sixth
!     s = thole(i)*thole(j)
      
      If (IScreen.eq.0) then
         lambda3 = one
         lambda5 = one   
      Else If (IScreen.eq.1) then
         ! Compute screening
         s = thole(i)*thole(j)
         
         ! Linear thole screening
         If ( R.lt.s ) then
           ! Compute the two screening factors
           v = R / s
           v3 = v**3
           lambda5 = v3*v 
           lambda3 = four*v3 - three*lambda5
         Else
           lambda3 = one
           lambda5 = one
         EndIf
      Else If (IScreen.eq.2) then
         ! Compute screening
         s = thole(i)*thole(j)

         ! Exponential thole screening
         v = R/s
         fExp = -v**3
         If (fExp .gt. -50.0d0) then
           ef = exp(fExp)
           lambda3 = one - ef
           lambda5 = one - (one-fExp)*ef
         Else
           lambda3 = one
           lambda5 = one
         EndIf
      EndIf
 
      rm3 = lambda3/r3
      rm5 = three*lambda5/r5
      ! Compute the scalar product between the dipole and the distance
      scd = mu(1,J)*RVec(1) + mu(2,J)*RVec(2) + mu(3,J)*RVec(3)

      term3(:) = mu(:,J)*rm3
      term5(:) = scd*rm5*rvec(:)

      efi = efi + term3 - term5

  EndDo
  E(:,I) = efi
EndDo

end subroutine TMu


logical function DoInter(I,J,NNlist,NAt,NNMax)

  ! Look whether I is in the nearest-neighbors of J
  
  integer I, J, NAt, NNMax
  integer, dimension(NNMax,NAt)  :: NNlist
  !
  integer L
  
  DoInter = .true.

  If (I.eq.J) then
    DoInter = .false.
    return
  EndIf

  ! Search in all neighbors of I
  Do L = 1, NNMax
    If (NNlist(L,I).eq.J) then
      DoInter = .false.
      return
    EndIf
  EndDo


end function
