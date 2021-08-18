import java.util.LinkedList;
import java.util.concurrent.Semaphore;

public class Storage {

    static private LinkedList<Object> list = new LinkedList<Object>();
    static final Semaphore notFull = new Semaphore(10);
    static final Semaphore notEmpty = new Semaphore(0);
    static final Semaphore mutex = new Semaphore(1);

    public static void main(String []args)
    {
        new Thread(new produce()).start();
        new Thread(new consume()).start();
    }

    static class produce implements Runnable
    {
       public void run()
        {
            while (true)
            {
		try {
		notFull.acquire();
		mutex.acquire();
            	list.add(new Object());
            	System.out.println("【生产者" + Thread.currentThread().getName()
                    + "】生产一个产品，现库存" + list.size());
		}
        	catch (Exception e) {
            	e.printStackTrace();
        	} 
		finally {
            	mutex.release();
            	notEmpty.release();
		}
	    }
        }
    }

    static class consume implements Runnable
    {
    	public void run()
    	{
	    while (true)
 	    {
        	try {
		notEmpty.acquire();
		mutex.acquire();
            	list.remove();
            	System.out.println("【消费者" + Thread.currentThread().getName()
                    + "】消费一个产品，现库存" + list.size());
		}
	       	catch (Exception e) {
            	e.printStackTrace();
        	}
	       	finally {
		mutex.release();
		notFull.release();
		}
	    }
        }
    }
}
