!final_version.f90
PROGRAM final_version

    !!!!!!!!!!!!!!!!!!Programa para calcular las colisiones!!!!!!!!!!!
    !!  Calculo de colisiones en 2d con condiciones periodicas      !!
    !!                                                              !!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !*****************************************************************************
  !  Modified:
  !
  !    19 Jule 2021
  !
  !  Author:
  !
  !    Manuel Mayo León 
  !
!?? Cosas en azul
!** Cosas en verde


implicit none



    INTEGER:: i,j,k,l,m
    REAL(kind=8),ALLOCATABLE,DIMENSION(:,:):: r,v !vector con posiciones y velocidades
    REAL(kind=8),ALLOCATABLE,DIMENSION(:,:,:)::sumv ! suma de velocidades para cada colision
    ! REAL(kind=8),ALLOCATABLE,DIMENSION(:,:,:)::densz ! densidad a lo largo de z
    REAL(kind=8),ALLOCATABLE,DIMENSION(:,:)::densz ! densidad a lo largo de z
    REAL(kind=8),ALLOCATABLE,DIMENSION(:,:):: tmp !temperaturas en las dos direcciones del espacio
    REAL(kind=8),ALLOCATABLE,DIMENSION(:,:,:):: density !densidad para tiempo t en funcion de k
    REAL(kind=8),ALLOCATABLE,DIMENSION(:,:):: densityprom !densidad para tiempo t en funcion de k
    REAL(kind=8),ALLOCATABLE,DIMENSION(:):: denspromz, stdevz !densidad promedio en z en el tiempo y su desviacion estandar 
    REAL(kind=8)::temp,tempz,H,longy,sigma,epsilon,rho !temperaturas en "y" y en "z", altura vertical y anchura, sigma==tamaño de la particula, rho=density
    REAL(kind=8)::alpha, vp  !! coeficiente de restitucion y velocidad que se introduce a traves de la pared 
    LOGICAL :: boolean,granular,densz_bool
    REAL(kind=8)::tcol,colt !tiempo de colision, tiempo para comparar y tiempo inicial
    INTEGER::rep,iter,n,iseed !numero de repeticiones que se realizan (tiempo) y numero de iteraciones  (numero de copias)
    REAL(kind=8),ALLOCATABLE,DIMENSION(:)::rab,vab !distancias y velocidades relativas
    INTEGER,DIMENSION(2)::ni !particulas que colisionan
    INTEGER,ALLOCATABLE,DIMENSION(:)::colisiones!numero de colisiones, tiempos de relajacion por repeticiones
    REAL(kind=8)::bij,qij,discr,t !bij=(ri-rj)*(vi-vj), discr es el discriminante de la solucion de segundo grado, t=tiempo de colision
    REAL(kind=8),ALLOCATABLE,DIMENSION(:)::tiempos,deltas !tiempos de colision
    REAL(kind=8), parameter :: pi = 4 * atan (1.0_8)
    REAL(kind=8) :: num_onda
    !!! para deteminar el tiempo de cálculo
    REAL(kind=4):: start, finish
    character(len=10)::alfa,eps
    INTEGER::partz !discretizacion en z para calcular la densidad
    ! INTEGER:: tiempo_relajacion,
    ! REAL(kind=4):: suma_particulas=0.0
    !notar que consideramos KT=1
    !inicializamos variables
    temp=1.0d00
    ! tempz=0.d001*temp
    tempz=5.0*temp
    ! temp=1.d00
    ! tempz=5.d00
    
    sigma=1.0d00
    
    H=1.5*sigma
    ! H=1.3*sigma
    ! H=1.9*sigma
    n=500
    ! rho=0.06d00
    ! rho=0.015d00
    ! rho=0.1d00
    rho=0.03d00
    ! rho=0.2111d00


    
    epsilon=(H-sigma)/sigma
    longy=REAL(n,8)/(rho*(H-sigma))
    ! rep=50000000
    ! rep=550000 !para 1.9*sigma
    ! rep=7000000 ! para 1.3*sigma
    rep=22000000!para 1.5*sigma
    !factor 
    iter=1

    alpha=0.95
    ! alpha=1.0
    ! vp=0.001*temp
    vp=0.0001
    ! vp=0.0d0
    
!!particiones en las que divido el espacio en z para medir la densidad en el equilibrio 
    partz=8
! Determinamos el numero de onda

num_onda=2*pi/longy
    
    ALLOCATE(r(n,2),v(n,2),sumv(iter,rep,2),tmp(rep,2),rab(2),vab(2),colisiones(iter),tiempos(rep),deltas(rep))
    ALLOCATE(densz(iter,partz),denspromz(partz),stdevz(partz),density(iter,rep,2),densityprom (rep,2))

    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) '            MD 2D SIMULATION                 '
    write ( *, '(a)' ) '            FORTRAN90 version                '
    
    write ( *, '(a)' ) ' '
    write ( *, '(a,g14.6)' ) '  Temperature y axis = ', temp
    write ( *, '(a,g14.6)' ) '  Temperature z axis = ', tempz
    write ( *, '(a,i10)' ) '  number of steps = ', rep
    write ( *, '(a,i8)' ) &
      '  The number of iterations taken  = ', iter
    write ( *, '(a,i8)' ) '  N = ', n
    write ( *, '(a,g14.6)' ) '  diameter (sigma) = ', sigma
    write ( *, '(a,g14.6)' ) '  density (rho) = ', rho
    write ( *, '(a,g14.6)' ) '  epsilon = ', epsilon
    write ( *, '(a,g14.6)' ) '  alpha = ', alpha
    write ( *, '(a,g14.6)' ) '  v_p = ', vp
     
    write ( *, '(a)' ) ' '


    WRITE(alfa,'(F10.2)')   alpha
    WRITE(eps,'(F10.2)')   epsilon
    !!!! para guardar los valores de las posiciones y velocidades iniciales!!!!!!

    call save_initial_distribution()

  
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !! llamamos al generador de numeros aleatorios
    iseed=2312
    call dran_ini(iseed)
    !pongo el numero de colisiones a cero
    colisiones(:)=0.d00
    !inicializo el tiempo para calcular el tiempo de computo
    call cpu_time(start)
    !Abro los archivos en los que voy a guardar los valores de las temperaturas, velocidades, posiciones, etc...
    
    ! If granular eqv true, the particles lost energy on each collision (inelastic disks) 
    !else, the collision is elastic between particles 
    granular=.TRUE.
    ! granular=.FALSE.
    
    ! If boolean eqv false, the particle is confined between two rigid plates 
    !else, the lower plate is vibrating in a sawtooth way 
    boolean=.TRUE.
    ! boolean=.FALSE.


    ! If densz_bool eqv false, the density is not calculated
    !else, the density is calculated
    densz_bool=.TRUE.
    ! densz_bool=.FALSE.
    densz=0.0
    denspromz=0.0
    stdevz=0.0
    density=0
    densityprom=0
    ! tiempo_relajacion=0

    DO i=1,iter

       

        !inicializo los tiempos 
        t=0.0
        deltas(1)=0.0
        CALL inicializacion2d(n,longy,H,temp,tempz,r,v)
        !PRINT*,'temperatura en y', 0.5d00*sum(v(:,1)**2)/n, 'temperatura en z' , 0.5d00*sum(v(:,2)**2)/n
        DO j=1,rep
            ni=0
            colt = HUGE(1.0)
            DO k=1,n

                CALL calcular_tiempo_de_colision(k)


            END DO
         
            !para saber si los tiempos son negativos de nuevo
            IF (colt<0) THEN 
                WRITE(*,*) 'TIEMPOS NEGATIVOS'
                PRINT*,  'iteracion', j
                STOP 
            END IF 
            !!!Hacer avanzar las posiciones hasta el momento de colisión
            t=t+colt
            tiempos(j)=t

 
                 !obtenemos las densidades para todo t en el eje horizontal. r(:,1) :eje y. r(:,2) :eje z.  
               
            density(i,j,1)=sum(cos(num_onda*r(:,1)) )
            density(i,j,2)=sum(sin(num_onda*r(:,1)) )
    
         ! density(i,j,2)=sum(sin(num_onda*r(:,1)) )
           

            ! avanzo las posiciones hasta el momento de colisión
            CALL evolve_positions()
        

            !colision entre particulas  
            IF (ni(2)<=n .AND. ni(2)>0) THEN
             
            CALL collide(ni(1),ni(2))
            colisiones(i)=colisiones(i)+1
            
            !medimos solo las densidades cuando colisionan particulas entre si
            !medimos cuando el sistema esta en equilibrio térmico, bajo el criterio de que la diferencia entre temperaturas sea menor que 0.1
            
          
            IF (densz_bool .EQV. .TRUE.) THEN 
                if (abs(sum(v(:,2)**2)/n-sum(v(:,2)**2)/n)<=0.1  ) then
                    
                    CALL medir_densidad(i)

                end if 
            END IF

            ! print*,'particulas que colisionan', ni(1),ni(2)
            END IF



            !colision entre particula a y muro
            IF (ni(2)>n) THEN

                CALL wall_collide(ni(1),ni(2))            

                   

            !colisiones(i)=colisiones(i)+1
            END IF

           

            ! OBTENEMOS LOS VALORES DE LAS TEMPERATURA 
               DO l=1,2
                !    sumv(i,j,l)=0.5d00*sum(v(:,l)**2)/n
                sumv(i,j,l)=sum(v(:,l)**2)/n
               END DO

               !Obtenemos el tiempo medio entre colisiones
                IF (j==1) THEN
                   
                   deltas(j)=((2*(1+alpha)*sigma*epsilon*rho)/SQRT(pi))*(SQRT(sumv(i,j,1))*tiempos(j))
                
                
                ELSE  
                    
                    deltas(j)=((2*(1+alpha)*sigma*epsilon*rho)/SQRT(pi))*(tiempos(j)-tiempos(j-1))*SQRT(sumv(i,j,1))

                    ! DO l=1,j-1
                    !     deltas(j)=deltas(j)+deltas(l)
                    ! END DO 
                END IF

                    
              
             
               
         
                

            
  
        END DO


        !!!promediamos por el numero total de colisiones!!!!!!!!!!!!!!!!!!!!!!!!! 
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        densz(i,:)=densz(i,:)/real(colisiones(i))
       
       
        CALL superpuesto()
       

        ! Guardamos los valores de las velocidades para todas las trayectorias de forma consecutiva. Asi aumentamos la estadística

        ! OPEN(11,FILE='velocidad_' // trim(adjustl(alfa)) // '.txt',  FORM ='FORMATTED',STATUS='UNKNOWN',POSITION='APPEND'&
        ! ,ACTION='READWRITE')   
        !         DO l=1,n
        !             WRITE(11,*) v(l,1), v(l,2)
        !         END DO
        !     CLOSE(11)
       

        !PRINT*, "numero de colisiones en la iteración ",i ,":", colisiones(i)
    END DO
    
    
    !Calculamos la temperatura promedio 
    DO l=1,rep
        DO m=1,2
        tmp(l,m)=sum(sumv(:,l,m))/iter

        ! tmp(l,m)=2*tmp(l,m)/(temp+tempz)
        END DO
    END DO 

    ! Calculamos la densidad promedio para cada tiempo dado
    DO l=1,rep
        DO m=1,2
        densityprom(l,m)= sum(density(:,l,m))/iter
        ! tmp(l,m)=2*tmp(l,m)/(temp+tempz)
        END DO
    END DO  


    ! DO l=1,partz
    !     DO m=tiempo_relajacion,rep 
    !         denspromz(l)=denspromz(l)+sum(densz(:,m,l))/(iter*(rep-tiempo_relajacion))
    !     END DO 
    ! END DO 
IF (densz_bool .EQV. .TRUE.) THEN 
    call calc_densz_prom(iter,partz)
END IF
! ! Calculamos la densidad promedio junto con la desviación estándar en el eje z
!     DO l=1,partz

!             denspromz(l)=denspromz(l)+sum(densz(:,l))/(iter)

!             stdevz(l)=sqrt(sum(densz(:,l)**2)/iter-denspromz(l)**2)

!             Print*,"standard deviation", stdevz(l)

!     END DO 


! ! para determinar la densidad promedio y su desviación estándar en el eje z debemos finalmente dividir entre las secciones
! ! en las que hemos dividido el sistema para medir. 
!     denspromz(:)=denspromz(:)*real(partz)/(H-sigma)
!     stdevz(:)=stdevz(:)*real(partz)/(H-sigma)

    OPEN(9,FILE='temperaturas_' // trim(adjustl(alfa)) // '_' // trim(adjustl(eps)) // '.txt',STATUS='unknown')
    DO l=1,rep
        WRITE(9,*) tmp(l,1), tmp(l,2)
    END DO 
    CLOSE(9)   


    ! OPEN(16,FILE='densidad_horizontal_' // trim(adjustl(alfa)) // '_' // trim(adjustl(eps)) // '.txt',STATUS='unknown')
    ! DO l=1,rep
    !     WRITE(16,*) densityprom(l,1), densityprom(l,2)
    ! END DO 
    ! CLOSE(16)   

        

    OPEN(10,FILE='pos_' // trim(adjustl(alfa)) // '.txt',STATUS='unknown')   
        DO l=1,n
         WRITE(10,*) r(l,1), r(l,2)
        END DO
    CLOSE(10) 



    
    OPEN(12,FILE='tiemposdecol_' // trim(adjustl(alfa)) // '.txt',STATUS='unknown') 
    DO l=1,rep
        WRITE(12,*) tiempos(l)
    END DO
    CLOSE(12) 
    
    OPEN(13,FILE='sparam_' // trim(adjustl(alfa)) // '.txt',STATUS='unknown')
    DO l=1,rep
        WRITE(13,*) deltas(l)
    END DO
    CLOSE(13)



    OPEN(66,FILE='densitypromy_' // trim(adjustl(alfa)) // '.txt',STATUS='unknown')
    DO l=1,rep
        WRITE(66,*)   densityprom(l,1), densityprom(l,2) 
    END DO
    CLOSE(66)


 
   
    call save_data_file()
 
    !final del programa
    call cpu_time(finish)
    !calcula el tiempo de computo
    WRITE(*,*) '(Tiempo = ', finish-start , 'segundos.)'

    

   


    DEALLOCATE(r,v,sumv,tmp,rab,vab,colisiones,tiempos,deltas)



    CONTAINS
        !!!! para guardar los valores de las posiciones y velocidades iniciales!!!!!!
        subroutine save_initial_distribution()
          
            OPEN(7,FILE='velocidad_init.txt',STATUS='unknown')                       
            OPEN(8,FILE='posiciones_init.txt',STATUS='unknown')                      
            CALL inicializacion2d(n,longy,H,temp,tempz,r,v)                         
            DO i=1,n                                                                
                WRITE(7,*) v(i,1), v(i,2)
                WRITE(8,*)  r(i,1), r(i,2)
            END DO
            CLOSE(7)
            CLOSE(8)

        end subroutine save_initial_distribution


        subroutine save_data_file()
            implicit none 
            OPEN(UNIT=35,FILE='data.txt', FORM ='FORMATTED',STATUS='UNKNOWN',POSITION='APPEND'&
            ,ACTION='READWRITE')
    
            write(35,* ) 'N', n, 'T_y', temp, 'T_z', tempz, 'epsilon', epsilon,  ' alpha    '&
            , alpha, ' vp ', vp  , ' rho ', rho , ' colisiones p.p. ', colisiones(iter)/n , ' tiempo ', rep
            write(35,* ) ' '
          
        end subroutine save_data_file
   

        SUBROUTINE collide ( a, b)
            IMPLICIT NONE
            INTEGER, INTENT(in)  :: a, b   ! Colliding atom indices
            ! LOGICAL :: granular
            ! This routine implements collision dynamics, updating the velocities
            ! The colliding pair (i,j) is assumed to be in contact already

            REAL(kind=8), DIMENSION(2) :: rij, vij
            REAL(kind=8)               :: factor

            rij(:) = r(a,:) - r(b,:)
            rij(1) = rij(1) - longy*ANINT ( rij(1)/(longy) ) ! Separation vector
            vij(:) = v(a,:) - v(b,:)           ! Relative velocity

            factor = DOT_PRODUCT ( rij, vij )
            if(granular .EQV. .TRUE.) THEN 
                vij    = -((1.0d0+alpha)*factor * rij)/(2.0d0)
            ELSE
                vij    = -factor * rij
            END IF

            v(a,:) = v(a,:) + vij
            v(b,:) = v(b,:) - vij
                ! PRINT*, "velocidad particula 1",v(i,:)
                ! PRINT*, "velocidad particula 2,",v(j,:)
        END SUBROUTINE collide
    
        subroutine calcular_tiempo_de_colision(c) 
            IMPLICIT NONE

            INTEGER, INTENT(in)  :: c
            
            

            IF(c/=n) THEN    
                rab(:)=r(c,:)-r(c+1,:) ! calculamos posiciones relativas
                rab(1)=rab(1)-longy*ANINT(rab(1)/(longy)) ! condiciones periodicas
                vab(:)=v(c,:)-v(c+1,:)   !calculamos velocidades relativas
                bij    = DOT_PRODUCT ( rab, vab )   ! obtenemos el producto escalar (ri-rj)*(vi-vj)

                !! FIRST WAY TO COMPUTE 

                    IF (bij<0 ) THEN
                    discr=bij**2-(SUM(rab**2)-sigma**2)*SUM(vab**2)
                    IF( discr>0.0) THEN ! si colisiona con la sucesiva particula
                        ! tcol = ( -bij - SQRT ( discr ) ) / ( SUM ( vab**2 ) )
                    !! ALTERNATIVE WAY 
                        
                        qij=-(bij+sign(1.0d00,bij)*dsqrt(discr))    
                        !  
                        tcol=MIN(qij/abs(dsqrt(sum(vab**2))),(sum(rab**2)-sigma**2)/qij )
                !comprobar que los tiempos no son negativos
                        IF (tcol<0) THEN 
                        PRINT*, 'colisión:',c,c+1,'tiempo',tcol
                        END IF 
                        
                        IF (tcol<colt ) THEN
                            ! PRINT*, 'colisión:',k,k+1,'tiempo',tcol
                            colt=tcol
                            ni(1)=c
                            ni(2)=c+1
                        END IF
                    END IF
                    END IF
      
            END IF
            
            
            !!!!!!!!!!!!!!!! Si consideramos la partícula 1, vemos si esta colisiona con la última!!!!!!!!!!!!!!    
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!                    
            IF (c==1) THEN
                rab(:)=r(c,:)-r(n,:) ! calculamos posiciones relativas
                rab(1)=rab(1)-longy*ANINT(rab(1)/(longy)) ! condiciones periodicas
                vab(:)=v(c,:)-v(n,:)   !calculamos velocidades relativas
                bij    = DOT_PRODUCT ( rab, vab )   ! obtenemos el producto escalar (ri-rj)*(vi-vj)
                IF (bij<0.0d0 ) THEN
                discr=bij**2-(SUM(rab**2)-sigma**2)*SUM(vab**2)
                IF( discr>0.0d0) THEN ! si colisiona con la sucesiva particula
                    tcol = ( -bij - SQRT ( discr ) ) / ( SUM ( vab**2 ) )
                    
                    !comprobar que los tiempos no son negativos

                    IF (tcol<0.0d0) THEN 
                        PRINT*, 'colisión:',c,' con',n,'. Tiempo',tcol
                    END IF  
                    
                    IF (tcol<colt ) THEN
                        ! PRINT*, 'colisión:',k,' con',n,'. Tiempo',tcol
                        colt=tcol
                        ni(1)=c
                        ni(2)=n
                    END IF
                END IF
                END IF
    
    
    
            
        
        
            END IF

                
                !!!!!!!!!!!!!!!!!!!!!!!!!!! Colisión con las paredes !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



                IF (v(c,2)>0.0d0 ) THEN
                    tcol=(H-sigma*0.5d0-r(c,2))/v(c,2)

                    !comprobar que los tiempos no son negativos
                    IF (tcol<0) THEN 
                        PRINT*, 'colisión:',c,' con pared. Tiempo',tcol
                    END IF 
                
                        IF (tcol<colt ) THEN
                            colt=tcol
                            ni(1)=c
                            ni(2)=n+1
                        END IF
                END IF
                IF (v(c,2)<0.0d0 ) THEN
                    IF(boolean .EQV. .TRUE.) THEN 
                        tcol=(sigma*0.5d0-r(c,2))/(v(c,2)-vp)
                    ELSE
                        tcol=(sigma*0.5d0-r(c,2))/v(c,2)
                    END IF
                        !comprobar que los tiempos no son negativos
                        IF (tcol<0.0d0 ) THEN 
                            PRINT*, 'colisión:',k,' con pared. Tiempo',tcol
                        END IF 

                        IF (tcol<colt) THEN
                        colt=tcol
                        ni(1)=c
                        ni(2)=n+2
                        END IF
                END IF
                



        end subroutine calcular_tiempo_de_colision

        subroutine  evolve_positions()
            IMPLICIT NONE

            r(:,1)     = r(:,1) + colt * v(:,1) 
            r(:,2)     = r(:,2) + colt* v(:,2)   ! Advance all positions by t (box=1 units)
            r(:,1)     = r(:,1) - longy*ANINT(r(:,1)/(longy)) ! Apply periodic boundaries
        
        end subroutine evolve_positions
        !!!!! COLISIONES ENTRE UNA PARTICULA Y LA PARED. 
        ! granular=.FALSE. en el caso de colisiones con dos paredes rígidas
        ! granular =.TRUE. en el caso de que la pared de abajo sea de tipo diente de sierra 
        subroutine wall_collide(p,q)
            IMPLICIT NONE
            INTEGER :: p,q 
            ! LOGICAL :: boolean
            
            if (boolean .EQV. .FALSE.) THEN 
                v(p,2)=-v(p,2)
            ELSE 
                IF(q==n+1) THEN 
                    v(p,2)=-v(p,2)
                ELSE 
                    v(p,2)=2.0d00*vp-v(p,2)
                    ! print*, "collision with bottom wall"
                END IF 
            END IF
        end subroutine wall_collide

        subroutine medir_densidad(a)

            IMPLICIT NONE
            INTEGER :: w,y 
            INTEGER:: a
        

            DO w=1,partz
                ! Calculamos el número de particulas comprendidas en un intervalo
                    DO  y=1,n
                            IF (r(y,2)<(sigma/2.0d0+real(w)*(H-sigma)/real(partz)) .AND. &
                                r(y,2)>(sigma/2.0d0+real(w-1)*(H-sigma)/real(partz))) THEN 
                                    densz(a,w) = (densz(a,w)+1.0)
                            ! print*, 'densz', densz(i,j,l), 'para ', l, 'iteracion', j
                            END IF    
                    END DO
    
                END DO 
                

              

        end subroutine


        SUBROUTINE superpuesto() 
        LOGICAL::super
        INTEGER::q

        super=.FALSE. !! 1 si es falso, 0 si es verdadero
        iloop:DO q=1,n
            IF(ABS(r(q+1,1)-r(q,1)) <( 0.95) .AND. ABS(r(q+1,2)-r(q,2)) <( 0.95)) THEN
                !PRINT*, 'particula ',q,'con',q+1,'superpuestas',ABS(r(q+1,1)-r(q,1))
            super=.TRUE.
            !PRINT*, 'particula',q,'distansia', ABS(r(q+1,1)-r(q,1))
                EXIT iloop !salir del loop 
            END IF
        END DO iloop

        IF (super.EQV. .TRUE.) THEN 
        PRINT*, "LAS PARTICULAS ESTAN SUPERPUESTAS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! :,("
        ELSE
            PRINT*, "LAS PARTICULAS ESTAN BIEN :)"
        END IF

        END SUBROUTINE superpuesto

        subroutine calc_densz_prom(iteraciones,particion)

            IMPLICIT NONE
            INTEGER :: l,particion,iteraciones

            DO l=1,particion

                denspromz(l)=denspromz(l)+sum(densz(:,l))/(iteraciones)
    
                stdevz(l)=sqrt(sum(densz(:,l)**2)/iteraciones-denspromz(l)**2)
    
                Print*,"standard deviation", stdevz(l)
    
            END DO 
            ! para determinar la densidad promedio y su desviación estándar en el eje z debemos finalmente dividir entre las secciones
            ! en las que hemos dividido el sistema para medir. 
            denspromz(:)=denspromz(:)*real(partz)/(H-sigma)
            stdevz(:)=stdevz(:)*real(partz)/(H-sigma)

            OPEN(14,FILE='densz_' // trim(adjustl(alfa)) // '.txt',STATUS='unknown')
            DO l=1,particion
                WRITE(14,*) denspromz(l)   
            END DO
            CLOSE(14)
            OPEN(14,FILE='stdevz_' // trim(adjustl(alfa)) // '.txt',STATUS='unknown')
            DO l=1,partz
                WRITE(14,*) stdevz(l)    
            END DO
            CLOSE(14)

        end subroutine calc_densz_prom


END PROGRAM final_version